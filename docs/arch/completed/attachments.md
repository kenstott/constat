# Embedded Image Extraction from PDFs and Office Documents

## Goal

When ingesting PDFs, DOCX, PPTX, and XLSX files, extract embedded images, assign each a hierarchical address (`<data_source>:<parent_doc>:<image_name>`), and route them through the image processing pipeline (`images.md`) for vectorization.

This activates the existing `extract_images: bool` field on `DocumentConfig` which is currently a stub.

## Addressing Scheme

```
<data_source>:<parent_doc>:<image_name>
```

Where:
- `<data_source>` — the config key or `session` prefix (matches existing naming)
- `<parent_doc>` — the document name (may itself be a child, e.g., `inbox:msg_123`)
- `<image_name>` — derived from format-specific metadata or positional index

Examples:
```
# PDF images
business_rules:page_3_img_1.png
business_rules:page_7_img_2.jpg

# DOCX images
onboarding_guide:image1.png
onboarding_guide:image2.emf

# PPTX images
quarterly_deck:slide_4_img_1.png
quarterly_deck:slide_12_chart_1.png

# Email attachment containing a PDF with images
support-inbox:msg_20260301_abc:report.pdf:page_2_img_1.png

# Session-uploaded file
session:uploaded_report:page_1_img_1.png
```

The colon-separated address naturally composes with the email pipeline (`email.md`) — an image inside a PDF attached to an email gets a 4-level address.

## Extraction by Format

### PDF — `pypdf` + image extraction

`pypdf` provides `page.images` which yields embedded image objects.

```python
from pypdf import PdfReader
from PIL import Image
import io

def _extract_pdf_images(path: Path | None, data: bytes | None) -> list[ExtractedImage]:
    reader = PdfReader(path or io.BytesIO(data))
    images = []
    for page_num, page in enumerate(reader.pages, 1):
        for img_idx, image in enumerate(page.images, 1):
            name = f"page_{page_num}_img_{img_idx}"
            ext = _guess_extension(image.data[:8])  # magic bytes
            images.append(ExtractedImage(
                name=f"{name}{ext}",
                data=image.data,
                mime_type=_mime_from_ext(ext),
                page=page_num,
                index=img_idx,
            ))
    return images
```

### DOCX — `python-docx` image parts

```python
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT

def _extract_docx_images(path: Path | None, data: bytes | None) -> list[ExtractedImage]:
    doc = Document(path or io.BytesIO(data))
    images = []
    for idx, rel in enumerate(doc.part.rels.values(), 1):
        if rel.reltype == RT.IMAGE:
            image_part = rel.target_part
            filename = image_part.partname.split("/")[-1]  # e.g., "image1.png"
            images.append(ExtractedImage(
                name=filename,
                data=image_part.blob,
                mime_type=image_part.content_type,
                page=None,
                index=idx,
            ))
    return images
```

### PPTX — `python-pptx` slide images

```python
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

def _extract_pptx_images(path: Path | None, data: bytes | None) -> list[ExtractedImage]:
    prs = Presentation(path or io.BytesIO(data))
    images = []
    for slide_num, slide in enumerate(prs.slides, 1):
        img_idx = 0
        for shape in slide.shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                img_idx += 1
                image = shape.image
                ext = image.content_type.split("/")[-1]  # jpeg, png, etc.
                name = f"slide_{slide_num}_img_{img_idx}.{ext}"
                images.append(ExtractedImage(
                    name=name,
                    data=image.blob,
                    mime_type=image.content_type,
                    page=slide_num,
                    index=img_idx,
                ))
    return images
```

### XLSX — `openpyxl` embedded images

```python
from openpyxl import load_workbook

def _extract_xlsx_images(path: Path | None, data: bytes | None) -> list[ExtractedImage]:
    wb = load_workbook(path or io.BytesIO(data), data_only=True)
    images = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        for img_idx, image in enumerate(ws._images, 1):
            name = f"sheet_{sheet_name}_img_{img_idx}.png"
            img_data = image._data()
            images.append(ExtractedImage(
                name=name,
                data=img_data,
                mime_type="image/png",
                page=None,
                index=img_idx,
            ))
    return images
```

## Data Model

```python
@dataclass
class ExtractedImage:
    name: str               # filename for addressing
    data: bytes             # raw image bytes
    mime_type: str           # image/png, image/jpeg, etc.
    page: int | None        # page/slide number (PDF, PPTX)
    index: int              # positional index within page/document
```

## Integration with `_file_extractors.py`

Add extraction functions alongside existing text extractors. Each returns `list[ExtractedImage]`.

```python
def _extract_images_from_document(
    path: Path | None,
    data: bytes | None,
    doc_type: str,
) -> list[ExtractedImage]:
    """Dispatch to format-specific image extractor."""
    if doc_type in ("pdf", "application/pdf"):
        return _extract_pdf_images(path, data)
    elif doc_type in ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
        return _extract_docx_images(path, data)
    elif doc_type in ("pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"):
        return _extract_pptx_images(path, data)
    elif doc_type in ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"):
        return _extract_xlsx_images(path, data)
    return []
```

## Integration with `_core.py`

In `_extract_content()` or `_add_document_internal()`, after text extraction:

```python
# After extracting text content from the document...
content = _extract_text(result, doc_type)

# If extract_images is enabled, also extract embedded images
if doc_config.extract_images and doc_type not in ("image/",):
    embedded_images = _extract_images_from_document(result.path, result.data, doc_type)
    for img in embedded_images:
        img_name = f"{doc_name}:{img.name}"
        image_result = _extract_image(path=None, data=img.data)
        if image_result.category == "image-primary":
            desc = await self._describe_image(self._provider, img.data, img.mime_type)
            image_result.description = desc.description
            image_result.subcategory = desc.subcategory
            image_result.labels = desc.labels
        img_content = _render_image_result(image_result, img.name)
        self._loaded_documents[img_name] = LoadedDocument(
            name=img_name, content=img_content, doc_type="markdown",
        )
```

The extracted images flow into `_loaded_documents` and are indexed by the existing `_index_loaded_documents()` call — no changes to chunking, embedding, or storage.

## Interaction with Other Pipelines

### Email pipeline (`email.md`)

When an email attachment is a PDF/DOCX/PPTX, and both `extract_attachments` and `extract_images` are enabled:

```
support-inbox                                    # IMAP source
  → support-inbox:msg_123                        # email body
  → support-inbox:msg_123:report.pdf             # PDF attachment (text)
  → support-inbox:msg_123:report.pdf:page_2_img_1.png  # image inside PDF
```

The PDF attachment is first extracted by the email pipeline, then its embedded images are extracted by this pipeline. Address nesting composes naturally.

### Standalone image files (`images.md`)

Standalone image uploads (`.png`, `.jpg` directly) bypass this pipeline entirely — they go straight to `_extract_image()`. This pipeline only handles images *embedded within* other document formats.

## Configuration

The existing `extract_images: bool = False` field on `DocumentConfig` controls this feature. Default remains `False` to avoid unnecessary LLM vision calls and processing overhead.

```yaml
documents:
  onboarding_guide:
    path: ./docs/onboarding.pdf
    extract_images: true    # enable embedded image extraction
```

## Filtering

### Size filter

Skip images smaller than a threshold (icons, bullets, decorative elements):

```python
MIN_IMAGE_DIMENSION = 64    # pixels — skip if both width and height < 64
MIN_IMAGE_BYTES = 2048      # skip if raw data < 2KB
```

### Duplicate filter

PDFs and Office docs often embed the same image multiple times (headers, logos). Deduplicate by content hash:

```python
seen_hashes: set[str] = set()
for img in embedded_images:
    h = hashlib.sha256(img.data).hexdigest()[:16]
    if h in seen_hashes:
        continue
    seen_hashes.add(h)
    # process image...
```

## New Dependencies

None. All extraction uses libraries already in the project:
- `pypdf` — already used for PDF text extraction
- `python-docx` — already used for DOCX text extraction
- `python-pptx` — already used for PPTX text extraction
- `openpyxl` — already used for XLSX text extraction
- `Pillow` — already installed

## File Changes Summary

| File | Change |
|------|--------|
| `constat/discovery/doc_tools/_file_extractors.py` | Add `_extract_pdf_images()`, `_extract_docx_images()`, `_extract_pptx_images()`, `_extract_xlsx_images()`, `_extract_images_from_document()`, `ExtractedImage` |
| `constat/discovery/doc_tools/_core.py` | Wire image extraction in `_add_document_internal()` when `extract_images=True` |
| `tests/test_embedded_image_extraction.py` | Unit tests per format |

## Testing Strategy

1. **Unit tests per format**:
   - PDF with 2 embedded images → verify `_extract_pdf_images()` returns 2 `ExtractedImage` objects with correct page numbers
   - DOCX with 3 images → verify relationship traversal finds all image parts
   - PPTX with images on slides 1 and 4 → verify slide numbering in names
   - XLSX with chart screenshot → verify sheet name in image name

2. **Integration tests**:
   - Upload PDF with `extract_images: true` → verify parent text chunks + child image chunks both appear in vector store
   - Verify addresses: `doc_name:page_3_img_1.png` format
   - Verify deduplication: same logo on every page → indexed once

3. **Fixtures**: Create minimal test documents programmatically:
   - `tests/fixtures/doc_with_images.pdf` — 3-page PDF, image on page 2
   - `tests/fixtures/doc_with_images.docx` — DOCX with 2 embedded PNGs
   - `tests/fixtures/deck_with_images.pptx` — PPTX with image on slide 1

## Edge Cases

| Case | Handling |
|------|----------|
| Tiny images (icons, bullets) | Skip if below `MIN_IMAGE_DIMENSION` or `MIN_IMAGE_BYTES` |
| Duplicate images (logos, headers) | SHA-256 dedup within same parent document |
| EMF/WMF vector images (Office) | Skip — not rasterizable without system deps; log warning |
| CMYK JPEG (print PDFs) | Convert to RGB via Pillow before OCR/vision |
| Encrypted/password-protected PDF | `pypdf` raises — propagate error, skip image extraction |
| Hundreds of images in one doc | Process all but cap LLM vision calls at configurable limit (default 50) |

## Non-Goals (v1)

- Extracting charts/graphs as structured data (just treated as images)
- OCR of scanned PDF pages (handled by `images.md` when the full page is an image)
- Reconstructing image captions from surrounding text
- Image deduplication across different parent documents
