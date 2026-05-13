# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unit tests for EnhancedSearch wiring via _enhanced_adapter.

Verifies:
  - run_enhanced_search post-filters by chunk_type
  - MMR diversity: duplicate seed chunks are de-duplicated
  - Overfetch: limit is respected after filtering
  - Graceful no-op when no entity data (entity_index=None)
  - _RelationalEntityIndex adapter returns correct shape
  - _EnhancedStoreAdapter forwards domain_ids/session_id to vector.search
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Minimal DocumentChunk stand-in (avoids importing the real one)
# ---------------------------------------------------------------------------

@dataclass
class _FakeChunkType:
    value: str

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return self.value == getattr(other, "value", other)

    def __hash__(self):
        return hash(self.value)


@dataclass
class _FakeChunk:
    document_name: str
    content: str
    chunk_type: Any
    source: str = "document"
    section: str = ""
    chunk_index: int = 0


def _db_table_chunk(name: str) -> _FakeChunk:
    return _FakeChunk(document_name=f"schema:{name}", content=f"{name} schema", chunk_type=_FakeChunkType("db_table"), source="schema")


def _doc_chunk(name: str) -> _FakeChunk:
    return _FakeChunk(document_name=name, content=f"{name} content", chunk_type=_FakeChunkType("document"), source="document")


# ---------------------------------------------------------------------------
# _EnhancedStoreAdapter
# ---------------------------------------------------------------------------

class TestEnhancedStoreAdapter:

    def test_forwards_domain_ids_and_session_id(self):
        from constat.storage._enhanced_adapter import _EnhancedStoreAdapter
        import numpy as np

        vector = MagicMock()
        vector.search.return_value = []

        adapter = _EnhancedStoreAdapter(vector, domain_ids=["d1"], session_id="sess1")
        adapter.search(np.zeros(4), limit=3, query_text="test")

        vector.search.assert_called_once()
        call_kwargs = vector.search.call_args
        assert call_kwargs.kwargs.get("domain_ids") == ["d1"] or call_kwargs.args[2] == ["d1"]
        assert call_kwargs.kwargs.get("session_id") == "sess1" or call_kwargs.args[3] == "sess1"

    def test_vector_attribute_exposed(self):
        from constat.storage._enhanced_adapter import _EnhancedStoreAdapter
        vector = MagicMock()
        adapter = _EnhancedStoreAdapter(vector)
        assert adapter.vector is vector


# ---------------------------------------------------------------------------
# _RelationalEntityIndex
# ---------------------------------------------------------------------------

class TestRelationalEntityIndex:

    def _make_entity(self, eid: str):
        e = MagicMock()
        e.id = eid
        return e

    def test_get_entities_for_chunk_returns_tuples(self):
        from constat.storage._enhanced_adapter import _RelationalEntityIndex
        relational = MagicMock()
        relational.get_entities_for_chunk.return_value = [
            self._make_entity("e1"),
            self._make_entity("e2"),
        ]
        idx = _RelationalEntityIndex(relational, session_id="s1")
        result = idx.get_entities_for_chunk("chunk-abc")
        assert result == [("e1", 1.0), ("e2", 1.0)]

    def test_get_chunks_for_entity_returns_tuples(self):
        from constat.storage._enhanced_adapter import _RelationalEntityIndex
        relational = MagicMock()
        relational.get_chunks_for_entity.return_value = [
            ("cid1", MagicMock(), 0.9),
            ("cid2", MagicMock(), 0.7),
        ]
        idx = _RelationalEntityIndex(relational, session_id="s1")
        result = idx.get_chunks_for_entity("e1", top_n=2)
        assert result == [("cid1", 0.9), ("cid2", 0.7)]

    def test_get_entities_returns_empty_on_error(self):
        from constat.storage._enhanced_adapter import _RelationalEntityIndex
        relational = MagicMock()
        relational.get_entities_for_chunk.side_effect = RuntimeError("db error")
        idx = _RelationalEntityIndex(relational, session_id="s1")
        assert idx.get_entities_for_chunk("c1") == []

    def test_get_chunks_returns_empty_on_error(self):
        from constat.storage._enhanced_adapter import _RelationalEntityIndex
        relational = MagicMock()
        relational.get_chunks_for_entity.side_effect = RuntimeError("db error")
        idx = _RelationalEntityIndex(relational, session_id="s1")
        assert idx.get_chunks_for_entity("e1") == []


# ---------------------------------------------------------------------------
# run_enhanced_search
# ---------------------------------------------------------------------------

def _make_scored_chunk(chunk_id: str, chunk, score: float = 0.9):
    sc = MagicMock()
    sc.chunk_id = chunk_id
    sc.chunk = chunk
    sc.score = score
    return sc


class TestRunEnhancedSearch:
    import numpy as np

    def _vector(self):
        v = MagicMock()
        v.search.return_value = []
        v._conn = MagicMock()
        v.get_all_chunks.return_value = []
        return v

    def test_chunk_types_forwarded_to_store_adapter(self):
        """chunk_types is forwarded to the store adapter's search, not EnhancedSearch constructor.

        EnhancedSearch.__init__ has no chunk_types param; filtering happens via
        _EnhancedStoreAdapter which stores chunk_types and passes them to vector.search().
        """
        import numpy as np
        from constat.storage._enhanced_adapter import _EnhancedStoreAdapter

        vector_mock = MagicMock()
        vector_mock.search.return_value = []
        adapter = _EnhancedStoreAdapter(
            vector=vector_mock,
            domain_ids=None,
            session_id=None,
            chunk_types=["db_table"],
        )
        adapter.search(query_embedding=np.zeros(4), limit=5, query_text="customers")

        call_kwargs = vector_mock.search.call_args[1] if vector_mock.search.call_args else {}
        assert call_kwargs.get("chunk_types") == ["db_table"]

    def test_limit_passed_as_k_without_overfetch(self):
        """EnhancedSearch.search receives k == limit — no overfetch multiplier."""
        import numpy as np
        from constat.storage._enhanced_adapter import run_enhanced_search

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []

        with patch("constat.storage._enhanced_adapter.EnhancedSearch", return_value=mock_searcher):
            run_enhanced_search(
                vector=self._vector(),
                relational=None,
                query_embedding=np.zeros(4),
                limit=3,
                query_text=None,
                domain_ids=None,
                session_id=None,
                chunk_types=["db_table"],
            )

        call_k = mock_searcher.search.call_args[1].get("k") or mock_searcher.search.call_args[0][1]
        assert call_k == 3

    def test_no_chunk_type_filter_returns_all(self):
        """When chunk_types=None, all scored chunks are returned."""
        import numpy as np
        from constat.storage._enhanced_adapter import run_enhanced_search

        scored = [
            _make_scored_chunk("c1", _db_table_chunk("t1"), 0.9),
            _make_scored_chunk("c2", _doc_chunk("d1"), 0.8),
        ]

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = scored

        with patch("constat.storage._enhanced_adapter.EnhancedSearch", return_value=mock_searcher):
            results = run_enhanced_search(
                vector=self._vector(),
                relational=None,
                query_embedding=np.zeros(4),
                limit=5,
                query_text=None,
                domain_ids=None,
                session_id=None,
                chunk_types=None,
            )

        assert len(results) == 2

    def test_k_equals_limit_when_no_chunk_type_filter(self):
        """When chunk_types=None, k equals limit exactly."""
        import numpy as np
        from constat.storage._enhanced_adapter import run_enhanced_search

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []

        with patch("constat.storage._enhanced_adapter.EnhancedSearch", return_value=mock_searcher):
            run_enhanced_search(
                vector=self._vector(),
                relational=None,
                query_embedding=np.zeros(4),
                limit=7,
                query_text=None,
                domain_ids=None,
                session_id=None,
                chunk_types=None,
            )

        call_k = mock_searcher.search.call_args[1].get("k") or mock_searcher.search.call_args[0][1]
        assert call_k == 7

    def test_entity_index_built_when_relational_and_session_provided(self):
        """entity_index is passed to EnhancedSearch when relational + session_id given."""
        import numpy as np
        from constat.storage._enhanced_adapter import run_enhanced_search

        relational = MagicMock()
        relational.get_entities_for_chunk.return_value = []

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        captured = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return mock_searcher

        with patch("constat.storage._enhanced_adapter.EnhancedSearch", side_effect=_capture):
            run_enhanced_search(
                vector=self._vector(),
                relational=relational,
                query_embedding=np.zeros(4),
                limit=3,
                query_text=None,
                domain_ids=None,
                session_id="sess1",
                chunk_types=None,
            )

        assert captured.get("entity_index") is not None
        assert captured.get("entity_expansion") is True

    def test_entity_index_absent_when_no_session(self):
        """entity_index is None when no session_id — entity expansion disabled."""
        import numpy as np
        from constat.storage._enhanced_adapter import run_enhanced_search

        captured = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return MagicMock(search=MagicMock(return_value=[]))

        with patch("constat.storage._enhanced_adapter.EnhancedSearch", side_effect=_capture):
            run_enhanced_search(
                vector=self._vector(),
                relational=MagicMock(),
                query_embedding=np.zeros(4),
                limit=3,
                query_text=None,
                domain_ids=None,
                session_id=None,
                chunk_types=None,
            )

        assert captured.get("entity_index") is None
        assert captured.get("entity_expansion") is False
