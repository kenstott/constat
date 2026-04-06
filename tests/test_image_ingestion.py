# Copyright (c) 2025 Kenneth Stott
# Canary: 235172f2-34ab-4a41-8ab8-677792f69faa
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for image extraction and OCR pipeline."""

from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from constat.discovery.doc_tools._image import (
    OcrResult,
    ImageResult,
    _classify_image,
    _extract_image,
    _ocr_extract,
    _render_image_result,
    _describe_image_sync,
)
from constat.discovery.doc_tools._mime import detect_type_from_source


def _make_pytesseract_mock(data):
    """Create a mock pytesseract module with preset image_to_data return."""
    mock_tess = MagicMock()
    mock_tess.Output.DICT = "dict"
    mock_tess.image_to_data.return_value = data
    mock_tess.TesseractNotFoundError = type("TesseractNotFoundError", (Exception,), {})
    return mock_tess


class TestOcrExtract:
    def test_ocr_extract_basic(self):
        mock_image = MagicMock()
        mock_data = {
            "conf": [95, 90, -1, 85, 80],
            "text": ["Hello", "world", "", "foo", "bar"],
            "block_num": [1, 1, 1, 1, 1],
            "par_num": [1, 1, 1, 1, 1],
            "line_num": [1, 1, 1, 1, 1],
        }
        mock_tess = _make_pytesseract_mock(mock_data)
        with patch.dict("sys.modules", {"pytesseract": mock_tess}):
            result = _ocr_extract(mock_image)

        assert result.text == "Hello world foo bar"
        assert result.word_count == 4
        assert result.mean_confidence == pytest.approx(87.5)

    def test_ocr_extract_all_negative_conf(self):
        mock_image = MagicMock()
        mock_data = {"conf": [-1, -1], "text": ["", ""], "block_num": [1, 1], "par_num": [1, 1], "line_num": [1, 1]}
        mock_tess = _make_pytesseract_mock(mock_data)
        with patch.dict("sys.modules", {"pytesseract": mock_tess}):
            result = _ocr_extract(mock_image)

        assert result.mean_confidence == 0.0
        assert result.word_count == 0


class TestClassifyImage:
    def test_classify_text_primary(self):
        ocr = OcrResult(text="a " * 50, mean_confidence=80.0, word_count=50)
        assert _classify_image(ocr) == "text-primary"

    def test_classify_text_primary_boundary(self):
        ocr = OcrResult(text="a " * 50, mean_confidence=60.0, word_count=50)
        assert _classify_image(ocr) == "text-primary"

    def test_classify_image_primary_low_words(self):
        ocr = OcrResult(text="hello", mean_confidence=90.0, word_count=5)
        assert _classify_image(ocr) == "image-primary"

    def test_classify_image_primary_low_confidence(self):
        ocr = OcrResult(text="a " * 50, mean_confidence=50.0, word_count=50)
        assert _classify_image(ocr) == "image-primary"

    def test_classify_image_primary_both_low(self):
        ocr = OcrResult(text="", mean_confidence=0.0, word_count=0)
        assert _classify_image(ocr) == "image-primary"


class TestRenderImageResult:
    def test_render_text_primary(self):
        result = ImageResult(
            category="text-primary",
            subcategory="screenshot",
            ocr_text="Extracted content here",
            ocr_confidence=85.0,
            ocr_word_count=60,
            description=None,
            labels=[],
            source_path="test.png",
            dimensions=(800, 600),
        )
        md = _render_image_result(result, "test.png")
        assert "# test.png" in md
        assert "Extracted content here" in md
        # text-primary rendering is minimal (no Type/Dimensions metadata)
        assert "**Type:**" not in md

    def test_render_image_primary_with_description(self):
        result = ImageResult(
            category="image-primary",
            subcategory="chart",
            ocr_text="",
            ocr_confidence=0.0,
            ocr_word_count=0,
            description="A bar chart showing sales",
            labels=["chart", "sales", "quarterly"],
            source_path="chart.png",
            dimensions=(1024, 768),
        )
        md = _render_image_result(result, "chart.png")
        assert "# Image: chart.png" in md
        assert "**Type:** image-primary" in md
        assert "## Description" in md
        assert "A bar chart showing sales" in md
        assert "## Labels" in md
        assert "chart, sales, quarterly" in md
        assert "## Extracted Text" not in md

    def test_render_no_optional_sections(self):
        result = ImageResult(
            category="image-primary",
            subcategory="other",
            ocr_text="",
            ocr_confidence=0.0,
            ocr_word_count=0,
            description=None,
            labels=[],
            source_path="<bytes>",
            dimensions=(640, 480),
        )
        md = _render_image_result(result, "unknown.png")
        assert "## Extracted Text" not in md
        assert "## Description" not in md
        assert "## Labels" not in md


class TestExtractImageFromBytes:
    @staticmethod
    def _make_png_bytes(width, height):
        """Create a real PNG image in memory."""
        from PIL import Image as PILImage
        import io as _io
        img = PILImage.new("RGB", (width, height), color="red")
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_extract_from_bytes(self):
        data = self._make_png_bytes(100, 100)

        with patch("constat.discovery.doc_tools._image._ocr_extract") as mock_ocr:
            mock_ocr.return_value = OcrResult(text="test", mean_confidence=30.0, word_count=1)
            result = _extract_image(data=data)

        assert result.category == "image-primary"
        assert result.source_path == "<bytes>"
        assert result.dimensions == (100, 100)

    def test_extract_requires_path_or_data(self):
        with pytest.raises(ValueError, match="Either path or data must be provided"):
            _extract_image()

    def test_extract_downscales_large_image(self):
        # Use a real but large image
        data = self._make_png_bytes(8192, 4096)

        with patch("constat.discovery.doc_tools._image._ocr_extract") as mock_ocr:
            mock_ocr.return_value = OcrResult(text="", mean_confidence=0.0, word_count=0)
            result = _extract_image(data=data)

        assert max(result.dimensions) <= 4096
        assert result.dimensions == (4096, 2048)


class TestDescribeImageSync:
    def test_describe_parses_json(self):
        provider = MagicMock()
        provider.generate_vision.return_value = '{"description": "A photo", "subcategory": "photo", "labels": ["nature"]}'

        with patch("pathlib.Path.read_text", return_value="Describe this image"):
            result = _describe_image_sync(provider, b"imgdata", "image/png")

        assert result == {
            "description": "A photo",
            "subcategory": "photo",
            "labels": ["nature"],
        }
        provider.generate_vision.assert_called_once_with(
            system="You are an image analyst.",
            image_bytes=b"imgdata",
            mime_type="image/png",
            text_prompt="Describe this image",
        )

    def test_describe_json_parse_failure(self):
        provider = MagicMock()
        provider.generate_vision.return_value = "This is a nice photo of a sunset."

        with patch("pathlib.Path.read_text", return_value="Describe this image"):
            result = _describe_image_sync(provider, b"imgdata", "image/jpeg")

        assert result == {
            "description": "This is a nice photo of a sunset.",
            "subcategory": "other",
            "labels": [],
        }


class TestMimeImageDetection:
    def test_detect_image_from_mime(self):
        assert detect_type_from_source(None, "image/png") == "image"
        assert detect_type_from_source(None, "image/jpeg") == "image"
        assert detect_type_from_source(None, "image/tiff") == "image"
        assert detect_type_from_source(None, "image/webp") == "image"
        assert detect_type_from_source(None, "image/gif") == "image"

    def test_detect_image_from_extension(self):
        assert detect_type_from_source("/photos/pic.png", None) == "image"
        assert detect_type_from_source("/photos/pic.jpg", None) == "image"
        assert detect_type_from_source("/photos/pic.jpeg", None) == "image"
        assert detect_type_from_source("/photos/pic.tiff", None) == "image"
        assert detect_type_from_source("/photos/pic.gif", None) == "image"
        assert detect_type_from_source("/photos/pic.webp", None) == "image"
        assert detect_type_from_source("/photos/pic.bmp", None) == "image"
        assert detect_type_from_source("/photos/pic.svg", None) == "image"


class TestTesseractNotInstalled:
    def test_import_error_fallback(self):
        mock_image = MagicMock()
        with patch.dict("sys.modules", {"pytesseract": None}):
            # Force re-import to trigger ImportError
            import importlib
            import constat.discovery.doc_tools._image as img_mod
            # Directly test the function with pytesseract unavailable
            with patch("builtins.__import__", side_effect=_selective_import_error("pytesseract")):
                result = img_mod._ocr_extract(mock_image)

        assert result.text == ""
        assert result.mean_confidence == 0.0
        assert result.word_count == 0

    def test_tesseract_not_found_error(self):
        mock_image = MagicMock()
        mock_tess = _make_pytesseract_mock(None)
        mock_tess.image_to_data.side_effect = mock_tess.TesseractNotFoundError("not found")
        with patch.dict("sys.modules", {"pytesseract": mock_tess}):
            result = _ocr_extract(mock_image)

        assert result.text == ""
        assert result.mean_confidence == 0.0
        assert result.word_count == 0

    def test_os_error_fallback(self):
        mock_image = MagicMock()
        mock_tess = _make_pytesseract_mock(None)
        mock_tess.image_to_data.side_effect = OSError("tesseract not in PATH")
        with patch.dict("sys.modules", {"pytesseract": mock_tess}):
            result = _ocr_extract(mock_image)

        assert result.text == ""
        assert result.mean_confidence == 0.0
        assert result.word_count == 0


def _selective_import_error(blocked_module):
    """Return an __import__ replacement that raises ImportError only for blocked_module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{blocked_module}'")
        return real_import(name, *args, **kwargs)

    return _import


# ---------------------------------------------------------------------------
# Integration tests using data/images/ sample files
# ---------------------------------------------------------------------------

_IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"


class TestSampleImagePipeline:
    """End-to-end pipeline tests: file → _extract_image → _render_image_result → markdown."""

    @pytest.fixture(autouse=True)
    def require_images_dir(self):
        if not _IMAGES_DIR.exists():
            pytest.fail(f"data/images/ directory not found at {_IMAGES_DIR} — add test image fixtures")

    def test_scanned_text_extract(self):
        path = _IMAGES_DIR / "scanned_text.png"
        if not path.exists():
            pytest.fail("Test image file not found — add test fixtures")
        with patch("constat.discovery.doc_tools._image._ocr_extract") as mock_ocr:
            # Simulate OCR finding lots of text (text-primary)
            mock_ocr.return_value = OcrResult(
                text="QUARTERLY SALES REPORT Q4 2025 " + "word " * 60,
                mean_confidence=82.0,
                word_count=65,
            )
            result = _extract_image(path=path)

        assert result.category == "text-primary"
        assert result.dimensions[0] == 800
        assert result.dimensions[1] == 600
        assert result.source_path == str(path)

        md = _render_image_result(result, "scanned_text")
        assert "# Image: scanned_text" in md
        assert "## Extracted Text" in md
        assert "QUARTERLY SALES REPORT" in md
        assert len(md) > 100  # substantial content

    def test_photo_extract(self):
        path = _IMAGES_DIR / "photo.jpg"
        if not path.exists():
            pytest.fail("Test image file not found — add test fixtures")
        with patch("constat.discovery.doc_tools._image._ocr_extract") as mock_ocr:
            # Simulate OCR finding minimal text (image-primary)
            mock_ocr.return_value = OcrResult(text="", mean_confidence=0.0, word_count=0)
            result = _extract_image(path=path)

        assert result.category == "image-primary"
        assert result.dimensions == (1024, 768)

        md = _render_image_result(result, "photo")
        assert "# Image: photo" in md
        assert "image-primary" in md
        assert "## Extracted Text" not in md  # no text for image-primary

    def test_diagram_extract(self):
        path = _IMAGES_DIR / "diagram.png"
        if not path.exists():
            pytest.fail("Test image file not found — add test fixtures")
        with patch("constat.discovery.doc_tools._image._ocr_extract") as mock_ocr:
            # Simulate OCR finding some labels (image-primary, not enough words)
            mock_ocr.return_value = OcrResult(
                text="Client App API Gateway Auth Service Load Balancer Worker Database",
                mean_confidence=70.0,
                word_count=10,
            )
            result = _extract_image(path=path)

        assert result.category == "image-primary"  # only 10 words < 50 threshold

        md = _render_image_result(result, "diagram")
        assert "# Image: diagram" in md
        assert "image-primary" in md

    def test_photo_with_vision_description(self):
        """Simulate full pipeline: extract → LLM vision describe → render."""
        path = _IMAGES_DIR / "photo.jpg"
        if not path.exists():
            pytest.fail("Test image file not found — add test fixtures")

        with patch("constat.discovery.doc_tools._image._ocr_extract") as mock_ocr:
            mock_ocr.return_value = OcrResult(text="", mean_confidence=0.0, word_count=0)
            result = _extract_image(path=path)

        # Simulate LLM vision enrichment
        result.description = "A landscape with mountains under a blue sky with a yellow sun."
        result.subcategory = "photograph"
        result.labels = ["mountains", "landscape", "sky", "sun", "nature"]

        md = _render_image_result(result, "photo")
        assert "## Description" in md
        assert "mountains under a blue sky" in md
        assert "## Labels" in md
        assert "mountains, landscape, sky, sun, nature" in md

    def test_extract_content_image_branch(self):
        """Test the _extract_content method routes images correctly."""
        from constat.discovery.doc_tools._transport import FetchResult

        path = _IMAGES_DIR / "diagram.png"
        if not path.exists():
            pytest.fail("Test image file not found — add test fixtures")

        image_bytes = path.read_bytes()
        fetch_result = FetchResult(
            data=image_bytes,
            source_path=str(path),
            detected_mime="image/png",
        )

        # Create a minimal _CoreMixin instance to test _extract_content
        from constat.discovery.doc_tools._core import _CoreMixin
        mixin = object.__new__(_CoreMixin)
        mixin._router = None  # no vision LLM

        with patch("constat.discovery.doc_tools._image._ocr_extract") as mock_ocr:
            mock_ocr.return_value = OcrResult(text="labels here", mean_confidence=50.0, word_count=2)
            content, fmt = mixin._extract_content(fetch_result, "image")

        assert fmt == "markdown"
        assert "# Image: diagram" in content
        assert "image-primary" in content
