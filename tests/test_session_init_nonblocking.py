"""Tests for non-blocking document indexing during session init (Fix 1)."""

from unittest.mock import MagicMock
import pytest

from constat.server.models import EventType


class _FakeResources:
    def __init__(self, doc_names=None):
        self._doc_names = list(doc_names or [])

    @property
    def document_names(self):
        return self._doc_names

    def add_document(self, **kwargs):
        self._doc_names.append(kwargs["name"])


class _FakeDocTools:
    def __init__(self, success=True, msg="ok"):
        self._success = success
        self._msg = msg
        self.calls = []

    def add_document_from_config(self, name, doc_config, **kwargs):
        self.calls.append((name, doc_config, kwargs))
        return self._success, self._msg


def _make_managed(user_id="alice", doc_tools=None, existing_docs=None):
    managed = MagicMock()
    managed.user_id = user_id
    managed.session_id = "sess-1"
    managed.session.doc_tools = doc_tools
    managed.session.resources = _FakeResources(existing_docs)
    managed.session.config.documents = {}
    managed.session.schema_manager = MagicMock()
    managed.session.schema_manager.connections = {}
    managed._loaded_db_configs = {}
    managed._loaded_api_configs = {}
    return managed


def _make_resolved(documents=None, databases=None, apis=None):
    resolved = MagicMock()
    resolved.sources.documents = documents or {}
    resolved.sources.databases = databases or {}
    resolved.sources.apis = apis or {}
    return resolved


class TestApplyResolvedSkipDocuments:
    def test_skip_documents_skips_doc_loop(self):
        from constat.server.routes.sessions import _apply_resolved_source_overrides

        doc_tools = _FakeDocTools()
        managed = _make_managed(doc_tools=doc_tools)
        resolved = _make_resolved(documents={"imap-inbox": {"url": "imap://..."}})

        _apply_resolved_source_overrides(managed, resolved, skip_documents=True)

        assert len(doc_tools.calls) == 0

    def test_default_includes_documents(self):
        from constat.server.routes.sessions import _apply_resolved_source_overrides

        doc_tools = _FakeDocTools()
        managed = _make_managed(doc_tools=doc_tools)
        resolved = _make_resolved(documents={"imap-inbox": {"url": "imap://..."}})

        _apply_resolved_source_overrides(managed, resolved)

        assert len(doc_tools.calls) == 1
        assert doc_tools.calls[0][0] == "imap-inbox"

    def test_skip_documents_still_loads_dbs(self):
        from constat.server.routes.sessions import _apply_resolved_source_overrides

        managed = _make_managed()
        managed.session.schema_manager.connections = {}
        resolved = _make_resolved(
            databases={"mydb": {"type": "duckdb", "path": "/tmp/test.duckdb"}},
            documents={"imap-inbox": {"url": "imap://..."}},
        )

        _apply_resolved_source_overrides(managed, resolved, skip_documents=True)

        managed.session.schema_manager.add_database_dynamic.assert_called_once()

    def test_skip_documents_still_loads_apis(self):
        from constat.server.routes.sessions import _apply_resolved_source_overrides

        managed = _make_managed()
        resolved = _make_resolved(
            apis={"weather": {"base_url": "https://api.weather.com"}},
            documents={"imap-inbox": {"url": "imap://..."}},
        )

        _apply_resolved_source_overrides(managed, resolved, skip_documents=True)

        managed.session.add_domain_api.assert_called_once_with(
            "weather", {"base_url": "https://api.weather.com"}
        )


class TestIndexUserDocuments:
    def test_pushes_start_and_complete_events(self):
        from constat.server.routes.sessions import _index_user_documents

        doc_tools = _FakeDocTools(success=True, msg="ok")
        managed = _make_managed(doc_tools=doc_tools)
        resolved = _make_resolved(documents={"report": {"url": "https://example.com/report.pdf"}})
        sm = MagicMock()

        _index_user_documents(sm, managed, resolved)

        event_types = [c[0][1] for c in sm._push_event.call_args_list]
        assert EventType.SOURCE_INGEST_START in event_types
        assert EventType.SOURCE_INGEST_COMPLETE in event_types

    def test_error_pushes_error_event(self):
        from constat.server.routes.sessions import _index_user_documents

        doc_tools = MagicMock()
        doc_tools.add_document_from_config.side_effect = RuntimeError("IMAP timeout")
        managed = _make_managed(doc_tools=doc_tools)
        resolved = _make_resolved(documents={"email": {"url": "imap://..."}})
        sm = MagicMock()

        _index_user_documents(sm, managed, resolved)

        event_types = [c[0][1] for c in sm._push_event.call_args_list]
        assert EventType.SOURCE_INGEST_ERROR in event_types

    def test_triggers_ner_after_indexing(self):
        from constat.server.routes.sessions import _index_user_documents

        doc_tools = _FakeDocTools(success=True)
        managed = _make_managed(doc_tools=doc_tools)
        resolved = _make_resolved(documents={"doc1": {"url": "https://example.com"}})
        sm = MagicMock()

        _index_user_documents(sm, managed, resolved)

        sm.refresh_entities_async.assert_called_once_with(managed.session_id)

    def test_invalidates_ner_fingerprint_before_refresh(self):
        from constat.server.routes.sessions import _index_user_documents
        from constat.discovery.ner_fingerprint import _session_fingerprints

        doc_tools = _FakeDocTools(success=True)
        managed = _make_managed(doc_tools=doc_tools)
        resolved = _make_resolved(documents={"doc1": {"url": "https://example.com"}})
        sm = MagicMock()

        # Seed a cached fingerprint
        _session_fingerprints[managed.session_id] = "stale"

        _index_user_documents(sm, managed, resolved)

        assert managed.session_id not in _session_fingerprints

    def test_no_ner_when_no_docs_indexed(self):
        from constat.server.routes.sessions import _index_user_documents

        doc_tools = _FakeDocTools(success=False, msg="failed")
        managed = _make_managed(doc_tools=doc_tools)
        resolved = _make_resolved(documents={"doc1": {"url": "https://example.com"}})
        sm = MagicMock()

        _index_user_documents(sm, managed, resolved)

        sm.refresh_entities_async.assert_not_called()

    def test_skips_existing_docs(self):
        from constat.server.routes.sessions import _index_user_documents

        doc_tools = _FakeDocTools()
        managed = _make_managed(doc_tools=doc_tools, existing_docs=["already-there"])
        resolved = _make_resolved(documents={"already-there": {"url": "https://example.com"}})
        sm = MagicMock()

        _index_user_documents(sm, managed, resolved)

        assert len(doc_tools.calls) == 0

    def test_progress_callback_wired(self):
        from constat.server.routes.sessions import _index_user_documents

        doc_tools = _FakeDocTools(success=True)
        managed = _make_managed(doc_tools=doc_tools)
        resolved = _make_resolved(documents={"email": {"url": "imap://..."}})
        sm = MagicMock()

        _index_user_documents(sm, managed, resolved)

        assert len(doc_tools.calls) == 1
        kwargs = doc_tools.calls[0][2]
        assert "progress_callback" in kwargs
        assert callable(kwargs["progress_callback"])
