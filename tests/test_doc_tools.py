# Copyright (c) 2025 Kenneth Stott
# Canary: 6a396d74-c4eb-4d41-8169-9268f9f268c6
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Cross-format integration and error handling tests for doc_tools module."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from constat.core.config import Config
from constat.discovery.doc_tools import DocumentDiscoveryTools
from constat.discovery.doc_tools import _transport as _doc_transport
from constat.discovery.doc_tools._file_extractors import (
    _extract_docx_text_from_bytes,
    _extract_xlsx_text_from_bytes,
    _extract_pptx_text_from_bytes,
)

# Import helpers from split modules to avoid duplication
from tests.test_doc_tools_word import (
    create_minimal_docx,
    create_docx_with_content,
    create_minimal_pptx,
)
from tests.test_doc_tools_excel import (
    create_minimal_xlsx,
)


# Reset module-level HTTP session singleton between tests
@pytest.fixture(autouse=True)
def _reset_http_session():
    _doc_transport._http_session = None
    yield
    _doc_transport._http_session = None


# =============================================================================
# Cross-format Integration Tests
# =============================================================================


@pytest.mark.usefixtures("clear_document_embeddings")
class TestOfficeDocumentIntegration:
    """Integration tests for Office documents in the discovery workflow."""

    def test_list_documents_includes_office_docs(self, tmp_path):
        """Test that Office documents appear in document listing."""
        docx_path = tmp_path / "test.docx"
        docx_path.write_bytes(create_minimal_docx())
        xlsx_path = tmp_path / "test.xlsx"
        xlsx_path.write_bytes(create_minimal_xlsx())
        pptx_path = tmp_path / "test.pptx"
        pptx_path.write_bytes(create_minimal_pptx())

        config = Config(
            documents={
                "word_doc": {
                    "type": "docx",
                    "path": str(docx_path),
                    "description": "Word document",
                },
                "excel_doc": {
                    "type": "xlsx",
                    "path": str(xlsx_path),
                    "description": "Excel spreadsheet",
                },
                "ppt_doc": {
                    "type": "pptx",
                    "path": str(pptx_path),
                    "description": "PowerPoint presentation",
                },
            }
        )
        tools = DocumentDiscoveryTools(config)
        result = tools.list_documents()

        names = [doc["name"] for doc in result]
        assert "word_doc" in names
        assert "excel_doc" in names
        assert "ppt_doc" in names

    def test_office_content_searchable(self, tmp_path):
        """Test that Office document content can be searched."""
        docx_bytes = create_docx_with_content(paragraphs=["unique searchable keyword"])
        docx_path = tmp_path / "searchable.docx"
        docx_path.write_bytes(docx_bytes)

        config = Config(
            documents={
                "searchable_doc": {
                    "type": "docx",
                    "path": str(docx_path),
                }
            }
        )
        tools = DocumentDiscoveryTools(config)

        # First load the document
        tools.get_document("searchable_doc")

        # Then search
        result = tools.search_documents("unique searchable keyword", limit=5)

        assert len(result) > 0
        assert "searchable_doc" in result[0]["document"]

    def test_office_doc_loaded_only_once(self, tmp_path):
        """Test that Office documents are loaded once and cached."""
        docx_bytes = create_docx_with_content(paragraphs=["cached content"])
        docx_path = tmp_path / "cached.docx"
        docx_path.write_bytes(docx_bytes)

        config = Config(
            documents={
                "cached_doc": {
                    "type": "docx",
                    "path": str(docx_path),
                }
            }
        )
        tools = DocumentDiscoveryTools(config)

        # Load twice
        result1 = tools.get_document("cached_doc")
        result2 = tools.get_document("cached_doc")

        # Should be the same cached content
        assert result1["content"] == result2["content"]
        assert result1["loaded_at"] == result2["loaded_at"]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestOfficeDocumentErrorHandling:
    """Error handling tests for Office document loading."""

    def test_corrupt_docx_raises_error(self):
        """Test that corrupt DOCX data raises an error."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        corrupt_data = b"This is not a valid DOCX file"

        with pytest.raises(Exception):
            _extract_docx_text_from_bytes(corrupt_data)

    def test_corrupt_xlsx_raises_error(self):
        """Test that corrupt XLSX data raises an error."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        corrupt_data = b"This is not a valid XLSX file"

        with pytest.raises(Exception):
            _extract_xlsx_text_from_bytes(corrupt_data)

    def test_corrupt_pptx_raises_error(self):
        """Test that corrupt PPTX data raises an error."""
        config = Config(documents={})
        tools = DocumentDiscoveryTools(config)

        corrupt_data = b"This is not a valid PPTX file"

        with pytest.raises(Exception):
            _extract_pptx_text_from_bytes(corrupt_data)

    def test_http_office_fetch_error(self):
        """Test error handling when HTTP Office document fetch fails."""
        config = Config(
            documents={
                "http_doc": {
                    "type": "http",
                    "url": "https://example.com/document.docx",
                }
            }
        )
        tools = DocumentDiscoveryTools(config)

        mock_response = Mock()
        mock_response.raise_for_status = Mock(side_effect=Exception("404 Not Found"))

        with patch('constat.discovery.doc_tools._transport._get_http_session', return_value=Mock(get=Mock(return_value=mock_response))):
            result = tools.get_document("http_doc")

            assert "error" in result
            assert "Failed to load" in result["error"]
