# Copyright (c) 2025 Kenneth Stott
# Canary: 6a396d74-c4eb-4d41-8169-9268f9f268c6
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for PDF document loading in doc_tools module."""

from __future__ import annotations

import io
import re
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from constat.core.config import Config
from constat.discovery.doc_tools import DocumentDiscoveryTools
from constat.discovery.doc_tools import _transport as _doc_transport
from constat.discovery.doc_tools._file_extractors import (
    _extract_pdf_text,
    _extract_pdf_text_from_bytes,
)


# Reset module-level HTTP session singleton between tests
@pytest.fixture(autouse=True)
def _reset_http_session():
    _doc_transport._http_session = None
    yield
    _doc_transport._http_session = None


# =============================================================================
# Helper Functions for Creating Test PDFs
# =============================================================================


def create_minimal_pdf(num_pages: int = 1) -> bytes:
    """
    Create a minimal valid PDF file.

    Args:
        num_pages: Number of pages to create

    Returns:
        PDF file bytes
    """
    from pypdf import PdfWriter, PageObject

    writer = PdfWriter()
    for _ in range(num_pages):
        page = PageObject.create_blank_page(width=612, height=792)
        writer.add_page(page)

    output = io.BytesIO()
    writer.write(output)
    output.seek(0)
    return output.read()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pdf_reader():
    """Create a mock PdfReader for controlled testing."""
    def _create_mock(page_texts: list[str]):
        """
        Create a mock PdfReader with specified page texts.

        Args:
            page_texts: List of text content for each page (empty string for empty pages)
        """
        mock_reader = Mock()
        mock_pages = []

        for text in page_texts:
            mock_page = Mock()
            mock_page.extract_text.return_value = text
            mock_pages.append(mock_page)

        mock_reader.pages = mock_pages
        return mock_reader

    return _create_mock


@pytest.fixture
def simple_pdf_bytes():
    """Create a simple multi-page PDF."""
    return create_minimal_pdf(2)


@pytest.fixture
def pdf_file(tmp_path, simple_pdf_bytes):
    """Create a PDF file on disk."""
    pdf_path = tmp_path / "test_document.pdf"
    pdf_path.write_bytes(simple_pdf_bytes)
    return pdf_path


@pytest.fixture
def config_with_pdf_file(pdf_file):
    """Config with a PDF file document."""
    return Config(
        documents={
            "test_pdf": {
                "type": "file",
                "path": str(pdf_file),
                "description": "Test PDF document",
            }
        }
    )


@pytest.fixture
def config_with_pdf_type(pdf_file):
    """Config with a direct pdf type document."""
    return Config(
        documents={
            "direct_pdf": {
                "type": "pdf",
                "path": str(pdf_file),
                "description": "Direct PDF document",
            }
        }
    )


@pytest.fixture
def config_with_http_pdf():
    """Config with an HTTP PDF document."""
    return Config(
        documents={
            "http_pdf": {
                "type": "http",
                "url": "https://example.com/document.pdf",
                "description": "HTTP PDF document",
            }
        }
    )


@pytest.fixture
def config_with_pdf_url():
    """Config with a PDF type and URL."""
    return Config(
        documents={
            "url_pdf": {
                "type": "pdf",
                "url": "https://example.com/report.pdf",
                "description": "URL-based PDF document",
            }
        }
    )


# =============================================================================
# PDF Text Extraction Tests (using mocks for reliable testing)
# =============================================================================


class TestPdfTextExtraction:
    """Tests for _extract_pdf_text and _extract_pdf_text_from_bytes methods."""

    def test_extract_pdf_text_from_file_with_mock(self, tmp_path, mock_pdf_reader):
        """Test extracting text from a PDF file path."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        # Create a minimal PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_bytes = create_minimal_pdf(2)
        pdf_path.write_bytes(pdf_bytes)

        mock_reader = mock_pdf_reader(["Page one content.", "Page two content."])

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text(pdf_path)

        assert "[Page 1]" in result
        assert "[Page 2]" in result
        assert "Page one content" in result
        assert "Page two content" in result

    def test_extract_pdf_text_from_bytes_with_mock(self, mock_pdf_reader):
        """Test extracting text from PDF bytes."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(2)
        mock_reader = mock_pdf_reader(["First page text.", "Second page text."])

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text_from_bytes(pdf_bytes)

        assert "[Page 1]" in result
        assert "[Page 2]" in result
        assert "First page text" in result
        assert "Second page text" in result

    def test_page_markers_format(self, mock_pdf_reader):
        """Test that page markers follow expected format [Page N]."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(3)
        mock_reader = mock_pdf_reader(["A", "B", "C"])

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text_from_bytes(pdf_bytes)

        # Check page marker format
        page_markers = re.findall(r'\[Page \d+\]', result)
        assert len(page_markers) == 3
        assert page_markers[0] == "[Page 1]"
        assert page_markers[1] == "[Page 2]"
        assert page_markers[2] == "[Page 3]"

    def test_pages_separated_by_double_newline(self, mock_pdf_reader):
        """Test that pages are separated by double newlines."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(2)
        mock_reader = mock_pdf_reader(["Page 1", "Page 2"])

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text_from_bytes(pdf_bytes)

        # Pages should be joined by double newlines
        assert "\n\n" in result

    def test_empty_pages_are_skipped(self, mock_pdf_reader):
        """Test that empty pages are not included in output."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(4)
        # Pages 2 and 4 are empty
        mock_reader = mock_pdf_reader(["Content on page 1", "", "Content on page 3", ""])

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text_from_bytes(pdf_bytes)

        # Should have page 1 and page 3
        assert "[Page 1]" in result
        assert "[Page 3]" in result
        # Empty pages should be skipped
        assert "[Page 2]" not in result
        assert "[Page 4]" not in result

    def test_whitespace_only_pages_are_skipped(self, mock_pdf_reader):
        """Test that whitespace-only pages are treated as empty."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(3)
        # Page 2 has only whitespace
        mock_reader = mock_pdf_reader(["Content", "   \n\t  ", "More content"])

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text_from_bytes(pdf_bytes)

        assert "[Page 1]" in result
        assert "[Page 3]" in result
        assert "[Page 2]" not in result


class TestPdfErrorHandling:
    """Tests for error handling in PDF extraction."""

    def test_corrupt_pdf_raises_error(self):
        """Test that corrupt PDF data raises an error."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        corrupt_data = b"This is not a valid PDF file"

        with pytest.raises(Exception):  # pypdf raises various exceptions
            _extract_pdf_text_from_bytes(corrupt_data)

    def test_corrupt_pdf_file_raises_error(self, tmp_path):
        """Test that corrupt PDF file raises an error."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        corrupt_file = tmp_path / "corrupt.pdf"
        corrupt_file.write_bytes(b"Not a PDF")

        with pytest.raises(Exception):
            _extract_pdf_text(corrupt_file)

    def test_missing_pdf_file_raises_error(self):
        """Test that missing PDF file raises FileNotFoundError."""
        config = Config(
            documents={
                "missing": {
                    "type": "pdf",
                    "path": "/nonexistent/file.pdf",
                }
            }
        )
        tools = DocumentDiscoveryTools(config)

        result = tools.get_document("missing")

        assert "error" in result
        assert "not found" in result["error"].lower() or "Failed to load" in result["error"]


class TestPdfLoadingViaHttpType:
    """Tests for loading PDFs through type='http' with PDF content-type."""

    def test_load_pdf_via_http_content_type(self, config_with_http_pdf, mock_pdf_reader):
        """Test that PDFs are detected via HTTP content-type header."""
        mock_reader = mock_pdf_reader(["HTTP PDF content"])

        mock_response = Mock()
        mock_response.content = b"fake pdf bytes"
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = Mock()

        with patch('constat.discovery.doc_tools._transport._get_http_session', return_value=Mock(get=Mock(return_value=mock_response))), \
             patch('pypdf.PdfReader', return_value=mock_reader):
            tools = DocumentDiscoveryTools(config_with_http_pdf)
            result = tools.get_document("http_pdf")

            assert "content" in result
            assert "[Page 1]" in result["content"]
            assert result["format"] == "text"

    def test_load_pdf_via_http_url_extension(self, mock_pdf_reader):
        """Test that PDFs are detected via .pdf URL extension."""
        config = Config(
            documents={
                "url_pdf": {
                    "type": "http",
                    "url": "https://example.com/document.pdf",
                }
            }
        )
        mock_reader = mock_pdf_reader(["URL extension PDF content"])

        mock_response = Mock()
        mock_response.content = b"fake pdf bytes"
        mock_response.headers = {"content-type": "application/octet-stream"}  # Not PDF content-type
        mock_response.raise_for_status = Mock()

        with patch('constat.discovery.doc_tools._transport._get_http_session', return_value=Mock(get=Mock(return_value=mock_response))), \
             patch('pypdf.PdfReader', return_value=mock_reader):
            tools = DocumentDiscoveryTools(config)
            result = tools.get_document("url_pdf")

            assert "content" in result
            assert "[Page 1]" in result["content"]
            assert result["format"] == "text"

    def test_http_pdf_with_custom_headers(self, mock_pdf_reader):
        """Test that custom headers are passed when fetching HTTP PDFs."""
        config = Config(
            documents={
                "auth_pdf": {
                    "type": "http",
                    "url": "https://example.com/secure.pdf",
                    "headers": {
                        "Authorization": "Bearer token123",
                        "X-Custom": "value",
                    }
                }
            }
        )
        mock_reader = mock_pdf_reader(["Secure PDF content"])

        mock_response = Mock()
        mock_response.content = b"fake pdf bytes"
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = Mock()

        mock_session_get = Mock(return_value=mock_response)
        mock_session = Mock(get=mock_session_get)
        with patch('constat.discovery.doc_tools._transport._get_http_session', return_value=mock_session), \
             patch('pypdf.PdfReader', return_value=mock_reader):
            tools = DocumentDiscoveryTools(config)
            result = tools.get_document("auth_pdf")

            # Verify headers were passed
            call_kwargs = mock_session_get.call_args[1]
            assert call_kwargs["headers"]["Authorization"] == "Bearer token123"
            assert call_kwargs["headers"]["X-Custom"] == "value"
            assert "content" in result

    def test_http_pdf_fetch_error(self, config_with_http_pdf):
        """Test error handling when HTTP PDF fetch fails."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock(side_effect=Exception("404 Not Found"))

        with patch('constat.discovery.doc_tools._transport._get_http_session', return_value=Mock(get=Mock(return_value=mock_response))):
            tools = DocumentDiscoveryTools(config_with_http_pdf)
            result = tools.get_document("http_pdf")

            assert "error" in result
            assert "Failed to load" in result["error"]


# =============================================================================
# Multi-page PDF Tests
# =============================================================================


class TestMultiPagePdf:
    """Tests for multi-page PDF handling."""

    def test_multi_page_pdf_all_pages_extracted(self, mock_pdf_reader):
        """Test that all pages are extracted from multi-page PDF."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(5)
        mock_reader = mock_pdf_reader([
            "Content on page 1",
            "Content on page 2",
            "Content on page 3",
            "Content on page 4",
            "Content on page 5",
        ])

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text_from_bytes(pdf_bytes)

        for i in range(1, 6):
            assert f"[Page {i}]" in result
            assert f"Content on page {i}" in result

    def test_page_order_preserved(self, mock_pdf_reader):
        """Test that page order is preserved in output."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(3)
        mock_reader = mock_pdf_reader([
            "MARKER_ALPHA",
            "MARKER_BETA",
            "MARKER_GAMMA",
        ])

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text_from_bytes(pdf_bytes)

        # Check that markers appear in correct order
        pos_alpha = result.find("MARKER_ALPHA")
        pos_beta = result.find("MARKER_BETA")
        pos_gamma = result.find("MARKER_GAMMA")

        assert pos_alpha < pos_beta < pos_gamma


# =============================================================================
# Edge Cases
# =============================================================================


class TestPdfEdgeCases:
    """Edge case tests for PDF handling."""

    def test_single_page_pdf(self, mock_pdf_reader):
        """Test handling of single-page PDF."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(1)
        mock_reader = mock_pdf_reader(["Only page content"])

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text_from_bytes(pdf_bytes)

        assert "[Page 1]" in result
        assert "Only page content" in result
        # Should not have page 2
        assert "[Page 2]" not in result

    def test_pdf_with_special_characters(self, mock_pdf_reader):
        """Test PDF with special characters."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(1)
        mock_reader = mock_pdf_reader(["Special: < > & \" ' $ % @"])

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text_from_bytes(pdf_bytes)

        assert "Special:" in result

    def test_all_empty_pages_pdf(self, mock_pdf_reader):
        """Test PDF where all pages are empty."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(3)
        mock_reader = mock_pdf_reader(["", "", ""])

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text_from_bytes(pdf_bytes)

        # Should return empty string
        assert result == ""

    def test_pdf_with_none_text_extraction(self):
        """Test PDF where extract_text returns None."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(2)

        # Create mock where extract_text returns None
        mock_reader = Mock()
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = None
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Valid content"
        mock_reader.pages = [mock_page1, mock_page2]

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text_from_bytes(pdf_bytes)

        # Should only have page 2 (page 1 returned None)
        assert "[Page 1]" not in result
        assert "[Page 2]" in result
        assert "Valid content" in result

    def test_pdf_page_numbering_starts_at_one(self, mock_pdf_reader):
        """Test that page numbers start at 1, not 0."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(1)
        mock_reader = mock_pdf_reader(["First page"])

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text_from_bytes(pdf_bytes)

        assert "[Page 1]" in result
        assert "[Page 0]" not in result

    def test_large_pdf_many_pages(self, mock_pdf_reader):
        """Test PDF with many pages."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        num_pages = 100
        pdf_bytes = create_minimal_pdf(num_pages)
        mock_reader = mock_pdf_reader([f"Page {i} content" for i in range(1, num_pages + 1)])

        with patch('pypdf.PdfReader', return_value=mock_reader):
            result = _extract_pdf_text_from_bytes(pdf_bytes)

        # Verify first, middle, and last pages
        assert "[Page 1]" in result
        assert "[Page 50]" in result
        assert "[Page 100]" in result
        assert "Page 1 content" in result
        assert "Page 50 content" in result
        assert "Page 100 content" in result


# =============================================================================
# Document Config Variations
# =============================================================================


class TestDocumentConfigVariations:
    """Test various DocumentConfig settings for PDFs."""

    def test_pdf_with_description(self, tmp_path, mock_pdf_reader):
        """Test PDF document with description."""
        pdf_path = tmp_path / "described.pdf"
        pdf_path.write_bytes(create_minimal_pdf(1))

        config = Config(
            documents={
                "described_pdf": {
                    "type": "pdf",
                    "path": str(pdf_path),
                    "description": "Important business document",
                }
            }
        )
        tools = DocumentDiscoveryTools(config)

        # Check listing
        docs = tools.list_documents()
        assert docs[0]["description"] == "Important business document"

    def test_pdf_with_tags(self, tmp_path, mock_pdf_reader):
        """Test PDF document with tags."""
        pdf_path = tmp_path / "tagged.pdf"
        pdf_path.write_bytes(create_minimal_pdf(1))

        config = Config(
            documents={
                "tagged_pdf": {
                    "type": "pdf",
                    "path": str(pdf_path),
                    "tags": ["finance", "quarterly", "2024"],
                }
            }
        )
        tools = DocumentDiscoveryTools(config)

        # Check listing
        docs = tools.list_documents()
        assert "finance" in docs[0]["tags"]
        assert "quarterly" in docs[0]["tags"]
        assert "2024" in docs[0]["tags"]


# =============================================================================
# Real PDF Integration Test (without mocking)
# =============================================================================


class TestRealPdfExtraction:
    """Tests using real PDF files without mocking, for full integration coverage."""

    def test_extract_from_blank_pdf(self):
        """Test extracting from a PDF with no text content."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(2)

        # Blank pages should result in empty string
        result = _extract_pdf_text_from_bytes(pdf_bytes)

        # Should be empty or minimal
        assert isinstance(result, str)
        # Blank pages should be skipped
        assert "[Page" not in result or result.strip() == ""

    def test_pdf_reader_called_with_bytes_io(self, mock_pdf_reader):
        """Test that PdfReader is called with BytesIO wrapper for bytes input."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_bytes = create_minimal_pdf(1)
        mock_reader = mock_pdf_reader(["test content"])

        with patch('pypdf.PdfReader', return_value=mock_reader) as mock_class:
            _extract_pdf_text_from_bytes(pdf_bytes)

            # Verify PdfReader was called with a BytesIO object
            assert mock_class.call_count == 1
            call_args = mock_class.call_args[0][0]
            assert isinstance(call_args, io.BytesIO)

    def test_pdf_reader_called_with_path_for_file(self, tmp_path, mock_pdf_reader):
        """Test that PdfReader is called with Path for file input."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(create_minimal_pdf(1))
        mock_reader = mock_pdf_reader(["test content"])

        with patch('pypdf.PdfReader', return_value=mock_reader) as mock_class:
            _extract_pdf_text(pdf_path)

            # Verify PdfReader was called with the path
            assert mock_class.call_count == 1
            call_args = mock_class.call_args[0][0]
            assert call_args == pdf_path
