# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unit tests for _EntityMixin — verifies NER is removed and stubs behave correctly."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


def _make_mixin():
    from constat.discovery.doc_tools._entities import _EntityMixin

    class _Impl(_EntityMixin):
        def __init__(self):
            self._schema_entities = ["order", "customer"]
            self._openapi_operations = []
            self._openapi_schemas = []
            self._graphql_types = []
            self._graphql_fields = []
            self._stop_list = set()
            self._vector_store = MagicMock()

    return _Impl()


class TestEntityMixinStubs:
    def test_process_metadata_through_ner_is_noop(self):
        mixin = _make_mixin()
        mixin.process_metadata_through_ner([("orders_table", "order id customer reference")])
        mixin._vector_store.add_entities.assert_not_called()

    def test_process_metadata_empty_list_noop(self):
        mixin = _make_mixin()
        mixin.process_metadata_through_ner([])
        mixin._vector_store.add_entities.assert_not_called()

    def test_extract_entities_for_session_returns_zero(self):
        mixin = _make_mixin()
        result = mixin.extract_entities_for_session(
            session_id="s1",
            domain_ids=["d1"],
            schema_entities=["order"],
        )
        assert result == 0
        mixin._vector_store.add_entities.assert_not_called()

    def test_set_schema_entities_updates_attribute(self):
        mixin = _make_mixin()
        mixin.set_schema_entities(["product", "invoice"])
        assert set(mixin._schema_entities) == {"product", "invoice"}

    def test_set_schema_entities_skips_if_unchanged(self):
        mixin = _make_mixin()
        original = mixin._schema_entities
        mixin.set_schema_entities(["order", "customer"])
        assert mixin._schema_entities is original

    def test_set_openapi_entities(self):
        mixin = _make_mixin()
        mixin.set_openapi_entities(["GET /orders"], ["OrderSchema"])
        assert mixin._openapi_operations == ["GET /orders"]
        assert mixin._openapi_schemas == ["OrderSchema"]

    def test_set_graphql_entities(self):
        mixin = _make_mixin()
        mixin.set_graphql_entities(["Order"], ["id", "name"])
        assert mixin._graphql_types == ["Order"]
        assert mixin._graphql_fields == ["id", "name"]
