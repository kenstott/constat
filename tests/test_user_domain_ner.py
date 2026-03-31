# Copyright (c) 2025 Kenneth Stott
# Canary: 4b35b51f-ee67-4b61-90c3-ccc4d893341f
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for user domain NER visibility (Fix 2)."""

from unittest.mock import MagicMock, patch
import pytest


class TestRunEntityExtractionIncludesUserId:
    def test_user_id_added_to_domain_ids(self):
        """_run_entity_extraction includes managed.user_id in domain_ids."""
        from constat.server.session_manager import SessionManager

        sm = SessionManager.__new__(SessionManager)
        sm._sessions = {}
        sm._lock = MagicMock()

        # Create mock managed session
        managed = MagicMock()
        managed.user_id = "alice"
        managed.active_domains = ["hr-reporting"]
        managed._dynamic_dbs = []
        managed.resolved_config = None

        # Create mock session
        session = MagicMock()
        session.config.domains = {"sales-analytics": MagicMock()}
        session.config.relationships = None
        session.config.ner_stop_list = None
        session.doc_tools = MagicMock()
        session.doc_tools._vector_store = None
        session.doc_tools._stop_list = set()
        session.schema_manager.get_entity_names.return_value = []
        session._get_api_entity_names.return_value = []

        sm._sessions["sess-1"] = managed

        # Patch extract_entities_for_session to capture args
        captured_domain_ids = []
        original = session.doc_tools.extract_entities_for_session

        def capture_extract(*args, **kwargs):
            captured_domain_ids.extend(kwargs.get("domain_ids", []))
            return {"added": [], "removed": []}

        session.doc_tools.extract_entities_for_session = capture_extract

        sm._run_entity_extraction("sess-1", session)

        assert "alice" in captured_domain_ids, f"Expected 'alice' in {captured_domain_ids}"
        assert "hr-reporting" in captured_domain_ids
        assert "sales-analytics" in captured_domain_ids


class TestChunkVisibilityFilterIncludesUserDomain:
    def test_user_domain_in_filter(self):
        """chunk_visibility_filter includes user_id in WHERE clause."""
        from constat.storage.duckdb_backend import DuckDBVectorBackend

        clause, params = DuckDBVectorBackend.chunk_visibility_filter(
            domain_ids=["hr-reporting", "alice"]
        )

        assert "alice" in params
        assert "hr-reporting" in params
        assert "IN" in clause


class TestBuildDomainMapsUserSource:
    def test_user_source_maps_to_user_id(self):
        """_build_domain_maps includes user: prefixed sources."""
        from constat.server.entity_state import _build_domain_maps

        config = MagicMock()
        config.domains = {}
        config.databases = {}
        config.apis = None
        config.documents = None

        session = MagicMock()
        doc_info = MagicMock()
        doc_info.source = "user:alice"
        session.resources.databases = {}
        session.resources.apis = {}
        session.resources.documents = {"imap-inbox": doc_info}

        _, source_to_domain = _build_domain_maps(config, session)
        assert source_to_domain["imap-inbox"] == "alice"

    def test_domain_source_still_works(self):
        """_build_domain_maps still handles domain: prefixed sources."""
        from constat.server.entity_state import _build_domain_maps

        config = MagicMock()
        config.domains = {}
        config.databases = {}
        config.apis = None
        config.documents = None

        session = MagicMock()
        doc_info = MagicMock()
        doc_info.source = "domain:hr-reporting"
        session.resources.databases = {}
        session.resources.apis = {}
        session.resources.documents = {"hr-docs": doc_info}

        _, source_to_domain = _build_domain_maps(config, session)
        assert source_to_domain["hr-docs"] == "hr-reporting"
