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

from __future__ import annotations
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


class TestLoadPersonalAccountsRegistersResource:
    def test_personal_account_registered_with_user_source(self):
        """_load_personal_accounts registers documents with source=f'user:{user_id}'
        so _build_domain_maps can map them to the user UUID for glossary display.

        Regression test for #12: domain_id='personal' was passed to add_document()
        which does not accept domain_id — causing TypeError, resource not registered,
        and user entities losing their '(user)' bucket in the glossary.
        """
        from constat.core.resources import SessionResources
        from constat.server.routes.sessions import _load_personal_accounts

        user_id = "alice-uuid"

        # Build a real SessionResources so add_document type errors are caught
        resources = SessionResources()

        acct = MagicMock()
        acct.active = True
        acct.display_name = "My Calendar"
        acct.type = "calendar"

        managed = MagicMock()
        managed.session_id = "sess-1"
        managed.user_id = user_id
        managed.session.config.data_dir = None
        managed.session.doc_tools.add_document_from_config.return_value = (True, "ok")
        managed.session.resources = resources

        sm = MagicMock()

        with patch(
            "constat.server.accounts.load_user_accounts",
            return_value={"my_calendar": acct},
        ), patch(
            "constat.server.accounts.account_to_document_config",
            return_value={"type": "calendar", "description": "My Calendar"},
        ), patch(
            "constat.core.source_config.DocumentConfig",
        ) as MockDocConfig:
            MockDocConfig.return_value = MagicMock()
            _load_personal_accounts(sm, managed, user_id)

        # The personal account must be registered in resources
        assert "my_calendar" in resources.documents, (
            "Personal account not registered in session resources"
        )
        doc_info = resources.documents["my_calendar"]
        assert doc_info.source == f"user:{user_id}", (
            f"Expected source='user:{user_id}', got source='{doc_info.source}'"
        )
