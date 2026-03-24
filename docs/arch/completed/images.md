# Image Ingestion Architecture

## Goal

Extend the document ingestion pipeline (`discovery/doc_tools/`) to accept image files (PNG, JPEG, TIFF, WebP, BMP, GIF, SVG) alongside existing MD/PDF/Office formats. Extracted text and/or generated descriptions become the vectorization source, stored in the existing `embeddings` table.

## Supported Formats

| Extension | MIME | Notes |
|-----------|------|-------|
| `.png` | `image/png` | |
| `.jpg` / `.jpeg` | `image/jpeg` | |
| `.tiff` / `.tif` | `image/tiff` | Multi-page: treat each page as separate image |
| `.webp` | `image/webp` | |
| `.bmp` | `image/bmp` | |
| `.gif` | `image/gif` | First frame only (animated) |
| `.svg` | `image/svg+xml` | Rasterize to PNG before processing |

## Pipeline Overview

```
image file
  → _transport.fetch_document()          # existing, unchanged
  → _mime.detect_type_from_source()      # add image/* recognition
  → _extract_image(result) ──────────────────────────────────┐
      ├─ 1. OCR text extraction (Tesseract)                  │
      ├─ 2. Classify: text-primary vs image-primary          │
      ├─ 3. If image-primary → LLM vision summary           │
      ├─ 4. Structured ImageResult                           │
      └─ 5. Render as text content ─────────────────────────→│
  → _chunk_document(name, rendered_text)  # existing chunker │
  → model.encode(texts)                   # existing embedder│
  → vector_store.add_chunks()             # existing storage │
  → _extract_and_store_entities()         # existing NER     │
```

## Processing Steps

### Step 1: OCR Text Extraction

Library: `pytesseract` (wraps Tesseract OCR engine).

```python
import pytesseract
from PIL import Image

def _ocr_extract(image: Image.Image) -> OcrResult:
    """Extract text and confidence from image via Tesseract."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    # Aggregate word confidences (ignore -1 = non-text regions)
    confidences = [c for c in data["conf"] if c != -1]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    text = pytesseract.image_to_string(image).strip()
    return OcrResult(text=text, mean_confidence=mean_conf, word_count=len(confidences))
```

### Step 2: Classify — Text-Primary vs Image-Primary

A heuristic classifier based on OCR output, not an ML model. Keeps dependencies minimal.

```python
TEXT_PRIMARY_THRESHOLD = 50   # minimum word count
CONFIDENCE_THRESHOLD = 60.0   # minimum mean OCR confidence (0-100)

@dataclass
class ImageClassification:
    category: Literal["text-primary", "image-primary"]
    subcategory: str           # e.g., "scanned-document", "photograph", "diagram", "chart", "screenshot", "illustration", "art"
    ocr_text: str
    ocr_confidence: float
    ocr_word_count: int
    description: str | None    # populated only for image-primary
    labels: list[str]          # classification tags

def _classify_image(ocr: OcrResult) -> Literal["text-primary", "image-primary"]:
    if ocr.word_count >= TEXT_PRIMARY_THRESHOLD and ocr.mean_confidence >= CONFIDENCE_THRESHOLD:
        return "text-primary"
    return "image-primary"
```

**Rationale**: Scanned documents produce high word counts with decent confidence. Photographs produce few/no words or low-confidence noise.

### Step 3: LLM Vision Summary (Image-Primary Only)

For images classified as `image-primary`, send to the session's LLM provider with a vision prompt. Skip this step entirely for `text-primary` images (the OCR text is sufficient).

```python
async def _describe_image(
    provider: LLMProvider,
    image_bytes: bytes,
    mime_type: str,
) -> ImageDescription:
    """Get structured description + subcategory from vision LLM."""
    response = await provider.complete(
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64encode(image_bytes).decode()}},
                {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
            ],
        }],
        response_format=ImageDescription,  # structured output
    )
    return response.parsed
```

**Prompt** (`IMAGE_DESCRIPTION_PROMPT`):

```
Analyze this image and provide:
1. A detailed text description (2-4 sentences) suitable for search indexing.
2. A subcategory: one of "photograph", "diagram", "chart", "screenshot", "illustration", "art", "map", "form", "other".
3. A list of classification labels (3-8 tags) describing key subjects, objects, themes.

Respond as JSON: {"description": "...", "subcategory": "...", "labels": ["...", ...]}
```

For `text-primary` images, subcategory is always `"scanned-document"` and labels are extracted from OCR text via existing NER.

### Step 4: Structured Result

```python
@dataclass
class ImageResult:
    category: Literal["text-primary", "image-primary"]
    subcategory: str
    ocr_text: str
    ocr_confidence: float
    ocr_word_count: int
    description: str | None      # LLM-generated (image-primary only)
    labels: list[str]
    source_path: str
    dimensions: tuple[int, int]  # (width, height)
```

### Step 5: Render as Text for Vectorization

Produce a single text string from `ImageResult` that becomes the input to the existing `_chunk_document → encode → add_chunks` pipeline.

```python
def _render_image_result(result: ImageResult, name: str) -> str:
    parts = [f"# Image: {name}"]
    parts.append(f"Type: {result.subcategory}")
    parts.append(f"Dimensions: {result.dimensions[0]}x{result.dimensions[1]}")
    if result.labels:
        parts.append(f"Labels: {', '.join(result.labels)}")
    if result.category == "text-primary":
        parts.append(f"\n## Extracted Text\n\n{result.ocr_text}")
    else:
        if result.description:
            parts.append(f"\n## Description\n\n{result.description}")
        if result.ocr_text:
            parts.append(f"\n## Extracted Text (partial)\n\n{result.ocr_text}")
    return "\n".join(parts)
```

This rendered text is passed directly to the existing `_chunk_document()` method — no changes to the chunking, embedding, or storage layers.

## Integration Points

### `_file_extractors.py`

Add `_extract_image()` function alongside existing `_extract_pdf_text()`, etc.

```python
def _extract_image(path: Path | None, data: bytes | None) -> ImageResult:
    """Top-level image extraction: OCR → classify → optional LLM describe."""
```

This is a sync function for OCR/classification. The LLM vision call is async and called from `_core.py`.

### `_core.py` — `_extract_content()`

Add image branch:

```python
if doc_type.startswith("image/"):
    image_result = _extract_image(path=result.path, data=result.data)
    if image_result.category == "image-primary":
        description = await self._describe_image(self._provider, image_bytes, doc_type)
        image_result.description = description.description
        image_result.subcategory = description.subcategory
        image_result.labels = description.labels
    return _render_image_result(image_result, name)
```

### `_mime.py`

Add image MIME types to the known-type registry. Return `"image"` as the normalized type.

### `server/routes/files.py`

Add image extensions to the allowed document upload list:

```python
DOC_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".html", ".htm", ".pptx", ".xlsx",
                  ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp", ".gif", ".svg"}
```

### `embeddings` table

No schema change. Images produce standard `DocumentChunk` objects with:
- `source = "document"`
- `chunk_type = "document"` (could add `"image"` to `ChunkType` enum if filtering is needed later)
- `section` = subcategory (e.g., `"photograph"`, `"scanned-document"`)

### `DocumentConfig`

No changes needed. The existing `extract_images: bool` field on `DocumentConfig` is for extracting images *from within* PDFs/PPTX — orthogonal to standalone image ingestion.

## New Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `pytesseract` | Python Tesseract OCR wrapper | `pip install pytesseract` |
| `Pillow` | Image loading/manipulation | Already installed (used by matplotlib) |
| `tesseract` | OCR engine (system) | `brew install tesseract` / `apt install tesseract-ocr` |

No new Python ML models. The LLM vision call uses the session's existing provider (Anthropic Claude, which supports vision natively).

## New `ChunkType` Value (Optional)

If downstream code needs to distinguish image-sourced chunks:

```python
class ChunkType(str, Enum):
    ...
    IMAGE = "image"
```

Add to `embeddings` table queries that filter by `chunk_type` if needed. Not strictly required — `section` field already carries the subcategory.

## File Changes Summary

| File | Change |
|------|--------|
| `constat/discovery/doc_tools/_file_extractors.py` | Add `_extract_image()`, `_ocr_extract()`, `_classify_image()`, `_render_image_result()` |
| `constat/discovery/doc_tools/_core.py` | Add image branch in `_extract_content()`, add `_describe_image()` async method |
| `constat/discovery/doc_tools/_mime.py` | Register image MIME types |
| `constat/server/routes/files.py` | Add image extensions to `DOC_EXTENSIONS` |
| `constat/core/models.py` | Add `ImageResult`, `OcrResult`, `ImageDescription` dataclasses |
| `constat/prompts/image_description.txt` | Vision prompt template |
| `tests/test_image_ingestion.py` | Unit tests for OCR, classification, rendering, end-to-end |
| `pyproject.toml` | Add `pytesseract` dependency |

## Testing Strategy

1. **Unit tests** (`test_image_ingestion.py`):
   - `_ocr_extract()` with a known scanned-text image → verify word count, confidence, text content
   - `_classify_image()` with high-word/high-confidence → `text-primary`; low-word → `image-primary`
   - `_render_image_result()` for both categories → verify markdown structure
   - `_extract_image()` end-to-end with fixture images

2. **Integration tests**:
   - Upload a scanned PDF page (as PNG) through `/documents/upload` → verify chunks appear in vector store with correct `section`
   - Upload a photograph → verify LLM vision call is made, description stored
   - Upload via `DocumentConfig(url="file://...", ...)` → verify transport + extraction works

3. **Fixtures**: Include 2-3 small test images in `tests/fixtures/`:
   - `scanned_text.png` — screenshot of text (~200 words)
   - `photo.jpg` — a photograph with minimal text
   - `diagram.png` — a technical diagram with some labels

## Edge Cases

| Case | Handling |
|------|----------|
| Corrupted/unreadable image | `Pillow` raises `UnidentifiedImageError` → propagate as extraction failure, log warning |
| Very large image (>20MP) | Downscale to max 4096px on longest edge before OCR (Tesseract performance) |
| Animated GIF | Extract first frame only via `image.seek(0)` |
| Multi-page TIFF | Iterate `ImageSequence.Iterator(image)`, process each page, concatenate results |
| SVG | Rasterize via `cairosvg.svg2png()` → then process as PNG. Add `cairosvg` as optional dep. |
| Zero OCR words | Classify as `image-primary`, proceed to LLM vision |
| LLM vision failure | Log warning, fall back to OCR-only result with `description=None` |
| No Tesseract installed | `pytesseract.TesseractNotFoundError` at import time → skip OCR, classify as `image-primary`, rely on LLM vision only |

## Non-Goals (v1)

- Extracting images embedded within PDFs/PPTX (existing `extract_images` flag — separate work)
- Image-to-image similarity search (would require CLIP embeddings, different vector space)
- Video frame extraction
- Handwriting recognition (Tesseract handles printed text; handwriting needs specialized models)
