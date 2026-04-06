# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unit tests for _build_documents URI handling in source_resolvers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


class _FakeDocConfig:
    def __init__(self, *, url=None, path=None, type=None, description=None):
        self.url = url
        self.path = path
        self.type = type
        self.description = description


class _FakeManagedBuildDocs:
    """Minimal managed stand-in for _build_documents."""

    def __init__(
        self,
        config_docs: dict | None = None,
        domain_docs: dict | None = None,
        file_refs: list | None = None,
    ):
        self.session = MagicMock()
        self.session.config.documents = config_docs or {}
        self.session.config.apis = {}
        self.active_domains = []
        self._dynamic_dbs = []
        self._dynamic_apis = []
        self._file_refs = file_refs or []
        self.resolved_config = None
        self.user_id = None

        if domain_docs is not None:
            self.active_domains = ["sales-analytics.yaml"]
            fake_domain = MagicMock()
            fake_domain.documents = domain_docs
            fake_domain.databases = {}
            fake_domain.apis = {}
            self.session.config.load_domain.return_value = fake_domain
        else:
            self.session.config.load_domain.return_value = None


class TestBuildDocumentsUri:
    def _call(self, managed):
        from constat.server.graphql.source_resolvers import _build_documents
        return _build_documents(managed)

    def test_config_doc_with_url_field(self):
        doc_cfg = _FakeDocConfig(url="https://example.com/wiki")
        managed = _FakeManagedBuildDocs(config_docs={"wiki": doc_cfg})
        docs = self._call(managed)
        assert len(docs) == 1
        assert docs[0].path == "https://example.com/wiki"

    def test_config_doc_with_path_field(self):
        doc_cfg = _FakeDocConfig(path="/data/rules.md")
        managed = _FakeManagedBuildDocs(config_docs={"rules": doc_cfg})
        docs = self._call(managed)
        assert len(docs) == 1
        assert docs[0].path == "/data/rules.md"

    def test_config_doc_url_takes_precedence_over_path(self):
        doc_cfg = _FakeDocConfig(url="https://example.com", path="/local.md")
        managed = _FakeManagedBuildDocs(config_docs={"mixed": doc_cfg})
        docs = self._call(managed)
        assert len(docs) == 1
        assert docs[0].path == "https://example.com"

    def test_domain_doc_with_url_field(self):
        doc_cfg = _FakeDocConfig(url="s3://bucket/key.pdf")
        managed = _FakeManagedBuildDocs(domain_docs={"s3doc": doc_cfg})
        docs = self._call(managed)
        assert len(docs) == 1
        assert docs[0].path == "s3://bucket/key.pdf"

    def test_session_ref_with_uri(self):
        ref = {
            "name": "webdoc",
            "uri": "https://example.com/doc",
            "has_auth": False,
            "description": "a web document",
            "added_at": "2025-01-01T00:00:00+00:00",
        }
        managed = _FakeManagedBuildDocs(file_refs=[ref])
        docs = self._call(managed)
        assert len(docs) == 1
        assert docs[0].path == "https://example.com/doc"
