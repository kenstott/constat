"""Tests for progress callback and SOURCE_INGEST_PROGRESS event type (Fix 3)."""

import inspect
import pytest

from constat.server.models import EventType


class TestSourceIngestProgressEventType:
    def test_event_type_exists(self):
        assert hasattr(EventType, "SOURCE_INGEST_PROGRESS")
        assert EventType.SOURCE_INGEST_PROGRESS.value == "source_ingest_progress"


class TestAddDocumentFromConfigAcceptsProgressCallback:
    def test_progress_callback_parameter(self):
        from constat.discovery.doc_tools._core import _CoreMixin

        sig = inspect.signature(_CoreMixin.add_document_from_config)
        params = list(sig.parameters.keys())
        assert "progress_callback" in params

    def test_progress_callback_default_none(self):
        from constat.discovery.doc_tools._core import _CoreMixin

        sig = inspect.signature(_CoreMixin.add_document_from_config)
        param = sig.parameters["progress_callback"]
        assert param.default is None
