# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Image extraction and OCR for the document ingestion pipeline."""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

_MAX_DIMENSION = 4096


@dataclass
class OcrResult:
    text: str
    mean_confidence: float
    word_count: int


@dataclass
class ImageResult:
    category: Literal["text-primary", "image-primary"]
    subcategory: str
    ocr_text: str
    ocr_confidence: float
    ocr_word_count: int
    description: str | None
    labels: list[str]
    source_path: str
    dimensions: tuple[int, int]


def _ocr_extract(image: PILImage.Image) -> OcrResult:
    """Run OCR on a PIL image and return structured results."""
    try:
        import pytesseract
    except ImportError:
        return OcrResult(text="", mean_confidence=0.0, word_count=0)

    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except (pytesseract.TesseractNotFoundError, OSError):
        return OcrResult(text="", mean_confidence=0.0, word_count=0)

    confidences = [c for c in data["conf"] if c >= 0]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    words = [t for t in data["text"] if t.strip()]
    text = " ".join(words)

    return OcrResult(text=text, mean_confidence=mean_confidence, word_count=len(words))


def _classify_image(ocr: OcrResult) -> Literal["text-primary", "image-primary"]:
    """Classify image based on OCR results."""
    if ocr.word_count >= 50 and ocr.mean_confidence >= 60.0:
        return "text-primary"
    return "image-primary"


def _extract_image(
    path: Path | None = None,
    data: bytes | None = None,
) -> ImageResult:
    """Extract text and metadata from an image file or raw bytes."""
    from PIL import Image

    if path is not None:
        image = Image.open(path)
    elif data is not None:
        image = Image.open(io.BytesIO(data))
    else:
        raise ValueError("Either path or data must be provided")

    # Handle multi-frame TIFF
    if image.format == "TIFF" and getattr(image, "n_frames", 1) > 1:
        image.seek(0)

    # Handle animated GIF
    if image.format == "GIF" and getattr(image, "is_animated", False):
        image.seek(0)

    # Downscale if too large
    w, h = image.size
    if max(w, h) > _MAX_DIMENSION:
        scale = _MAX_DIMENSION / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        w, h = new_w, new_h

    ocr = _ocr_extract(image)
    category = _classify_image(ocr)

    return ImageResult(
        category=category,
        subcategory="other",
        ocr_text=ocr.text,
        ocr_confidence=ocr.mean_confidence,
        ocr_word_count=ocr.word_count,
        description=None,
        labels=[],
        source_path=str(path) if path else "<bytes>",
        dimensions=(w, h),
    )


def _render_image_result(result: ImageResult, name: str) -> str:
    """Render an ImageResult as markdown text."""
    parts = [
        f"# Image: {name}",
        "",
        f"**Type:** {result.category} | **Subcategory:** {result.subcategory}",
        f"**Dimensions:** {result.dimensions[0]}x{result.dimensions[1]}",
    ]

    if result.ocr_text:
        parts.append("")
        parts.append("## Extracted Text")
        parts.append("")
        parts.append(result.ocr_text)

    if result.description:
        parts.append("")
        parts.append("## Description")
        parts.append("")
        parts.append(result.description)

    if result.labels:
        parts.append("")
        parts.append("## Labels")
        parts.append("")
        parts.append(", ".join(result.labels))

    return "\n".join(parts)


def _describe_image_sync(provider, image_bytes: bytes, mime_type: str) -> dict:
    """Use a vision-capable LLM to describe an image."""
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "image_description.txt"
    prompt = prompt_path.read_text()

    response_text = provider.generate_vision(
        system="You are an image analyst.",
        image_bytes=image_bytes,
        mime_type=mime_type,
        text_prompt=prompt,
    )

    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_nl = text.index("\n") if "\n" in text else len(text)
        text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        result = json.loads(text)
        return {
            "description": result["description"],
            "subcategory": result["subcategory"],
            "labels": result["labels"],
        }
    except (json.JSONDecodeError, KeyError):
        return {
            "description": response_text.strip(),
            "subcategory": "other",
            "labels": [],
        }
