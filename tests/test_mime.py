# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for MIME type normalization and detection."""

import pytest

from constat.discovery.doc_tools._mime import (
    normalize_type,
    detect_type_from_source,
    is_binary_type,
)


class TestNormalizeType:
    def test_short_aliases(self):
        assert normalize_type("pdf") == "pdf"
        assert normalize_type("html") == "html"
        assert normalize_type("markdown") == "markdown"
        assert normalize_type("text") == "text"
        assert normalize_type("docx") == "docx"
        assert normalize_type("xlsx") == "xlsx"
        assert normalize_type("pptx") == "pptx"
        assert normalize_type("auto") == "auto"

    def test_full_mime_types(self):
        assert normalize_type("application/pdf") == "pdf"
        assert normalize_type("text/html") == "html"
        assert normalize_type("text/markdown") == "markdown"
        assert normalize_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document") == "docx"

    def test_case_insensitive(self):
        assert normalize_type("PDF") == "pdf"
        assert normalize_type("Text/HTML") == "html"

    def test_empty_returns_auto(self):
        assert normalize_type("") == "auto"

    def test_legacy_transport_types(self):
        assert normalize_type("file") == "auto"
        assert normalize_type("http") == "auto"
        assert normalize_type("inline") == "auto"
        assert normalize_type("confluence") == "auto"
        assert normalize_type("notion") == "auto"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown document type"):
            normalize_type("video/mp4")


class TestDetectTypeFromSource:
    def test_mime_priority(self):
        assert detect_type_from_source("/file.txt", "application/pdf") == "pdf"

    def test_partial_mime_match(self):
        assert detect_type_from_source(None, "application/pdf; charset=utf-8") == "pdf"
        assert detect_type_from_source(None, "text/html; charset=utf-8") == "html"

    def test_extension_fallback(self):
        assert detect_type_from_source("/docs/file.pdf", None) == "pdf"
        assert detect_type_from_source("/docs/file.md", None) == "markdown"
        assert detect_type_from_source("/docs/file.xlsx", None) == "xlsx"

    def test_text_fallback(self):
        assert detect_type_from_source(None, None) == "text"
        assert detect_type_from_source("/file.unknown", None) == "text"

    def test_office_mime_partial(self):
        assert detect_type_from_source(None, "application/vnd.openxmlformats-officedocument.wordprocessingml.document") == "docx"
        assert detect_type_from_source(None, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") == "xlsx"


class TestIsBinaryType:
    def test_binary_types(self):
        assert is_binary_type("pdf") is True
        assert is_binary_type("docx") is True
        assert is_binary_type("xlsx") is True
        assert is_binary_type("pptx") is True

    def test_text_types(self):
        assert is_binary_type("text") is False
        assert is_binary_type("html") is False
        assert is_binary_type("markdown") is False
        assert is_binary_type("csv") is False
