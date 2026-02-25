# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for hybrid BM25 + vector search with Reciprocal Rank Fusion."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from constat.discovery.models import DocumentChunk, ChunkType
from constat.discovery.vector_store import DuckDBVectorStore


@pytest.fixture
def store(tmp_path):
    """Create an isolated DuckDBVectorStore in a temp directory."""
    db_path = str(tmp_path / "test_vectors.duckdb")
    os.environ["CONSTAT_VECTOR_STORE_PATH"] = db_path
    s = DuckDBVectorStore(db_path=db_path)
    yield s
    s._db.close()
    os.environ.pop("CONSTAT_VECTOR_STORE_PATH", None)


def _make_chunks(texts: list[str], source: str = "document") -> tuple[list[DocumentChunk], np.ndarray]:
    """Create chunks with random embeddings."""
    chunks = []
    for i, text in enumerate(texts):
        chunks.append(DocumentChunk(
            document_name=f"doc_{i}",
            content=text,
            section="test",
            chunk_index=i,
            source=source,
            chunk_type=ChunkType.DOCUMENT,
        ))
    embeddings = np.random.randn(len(texts), DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    return chunks, embeddings


class TestRRFMerge:
    """Unit tests for _rrf_merge static method."""

    def test_rrf_merge_basic(self):
        """Two lists with overlapping results produce correct RRF scores."""
        c1 = DocumentChunk(document_name="a", content="a", section="", chunk_index=0, chunk_type=ChunkType.DOCUMENT)
        c2 = DocumentChunk(document_name="b", content="b", section="", chunk_index=0, chunk_type=ChunkType.DOCUMENT)
        c3 = DocumentChunk(document_name="c", content="c", section="", chunk_index=0, chunk_type=ChunkType.DOCUMENT)

        vector = [("id_a", 0.9, c1), ("id_b", 0.8, c2)]
        bm25 = [("id_b", 5.0, c2), ("id_c", 3.0, c3)]

        merged = DuckDBVectorStore._rrf_merge(vector, bm25, k=60)

        ids = [cid for cid, _, _ in merged]
        # id_b appears in both lists — should be ranked first
        assert ids[0] == "id_b"
        # All three should be present
        assert set(ids) == {"id_a", "id_b", "id_c"}

    def test_rrf_scores_normalized_0_to_1(self):
        """RRF scores should be in [0, 1] range."""
        c1 = DocumentChunk(document_name="a", content="a", section="", chunk_index=0, chunk_type=ChunkType.DOCUMENT)
        vector = [("id_a", 0.9, c1)]
        bm25 = [("id_a", 5.0, c1)]

        merged = DuckDBVectorStore._rrf_merge(vector, bm25, k=60)
        for _, score, _ in merged:
            assert 0.0 <= score <= 1.0

    def test_rrf_best_score_is_1(self):
        """Item ranked #1 in both lists gets normalized score of 1.0."""
        c1 = DocumentChunk(document_name="a", content="a", section="", chunk_index=0, chunk_type=ChunkType.DOCUMENT)
        vector = [("id_a", 0.9, c1)]
        bm25 = [("id_a", 5.0, c1)]

        merged = DuckDBVectorStore._rrf_merge(vector, bm25, k=60)
        assert merged[0][1] == pytest.approx(1.0)

    def test_rrf_disjoint_lists(self):
        """Disjoint results all get partial scores."""
        c1 = DocumentChunk(document_name="a", content="a", section="", chunk_index=0, chunk_type=ChunkType.DOCUMENT)
        c2 = DocumentChunk(document_name="b", content="b", section="", chunk_index=0, chunk_type=ChunkType.DOCUMENT)
        vector = [("id_a", 0.9, c1)]
        bm25 = [("id_b", 5.0, c2)]

        merged = DuckDBVectorStore._rrf_merge(vector, bm25, k=60)
        assert len(merged) == 2
        for _, score, _ in merged:
            assert score < 1.0


class TestFTSDirtyFlag:
    """Tests for lazy FTS index rebuild."""

    def test_fts_dirty_on_init(self, store):
        assert store._fts_dirty is True

    def test_fts_dirty_cleared_after_rebuild(self, store):
        chunks, embeddings = _make_chunks(["hello world"])
        store.add_chunks(chunks, embeddings)
        assert store._fts_dirty is True
        store._rebuild_fts_index()
        assert store._fts_dirty is False

    def test_fts_dirty_set_on_add(self, store):
        chunks, embeddings = _make_chunks(["hello world"])
        store.add_chunks(chunks, embeddings)
        store._rebuild_fts_index()
        assert store._fts_dirty is False
        chunks2, emb2 = _make_chunks(["another doc"])
        # Need unique doc names
        chunks2[0].document_name = "doc_unique"
        store.add_chunks(chunks2, emb2)
        assert store._fts_dirty is True

    def test_fts_dirty_set_on_clear(self, store):
        chunks, embeddings = _make_chunks(["hello world"])
        store.add_chunks(chunks, embeddings)
        store._rebuild_fts_index()
        store.clear()
        assert store._fts_dirty is True

    def test_fts_dirty_set_on_delete_by_document(self, store):
        chunks, embeddings = _make_chunks(["hello world"])
        store.add_chunks(chunks, embeddings)
        store._rebuild_fts_index()
        store.delete_by_document("doc_0")
        assert store._fts_dirty is True


class TestHybridSearch:
    """Integration tests for hybrid BM25 + vector search."""

    def test_hybrid_search_returns_bm25_matches(self, store):
        """BM25 should surface exact keyword matches that vector search might miss."""
        texts = [
            "The ACME corporation handles all widget manufacturing processes",
            "General overview of business operations and strategy documents",
            "Financial reporting quarterly results and revenue analysis",
        ]
        chunks, embeddings = _make_chunks(texts)
        store.add_chunks(chunks, embeddings)

        # Use a random query embedding (won't match anything well semantically)
        query_emb = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)

        # Without hybrid — pure vector, returns something
        results_vector = store.search(query_emb, limit=3)
        assert len(results_vector) == 3

        # With hybrid — should also return results (BM25 contributes)
        results_hybrid = store.search(query_emb, limit=3, query_text="ACME widget manufacturing")
        assert len(results_hybrid) == 3
        # The ACME chunk should rank higher in hybrid due to exact keyword match
        hybrid_ids = [cid for cid, _, _ in results_hybrid]
        # doc_0 contains "ACME" and "widget" and "manufacturing"
        acme_chunk_ids = [cid for cid, _, c in results_hybrid if "ACME" in c.content]
        assert len(acme_chunk_ids) > 0

    def test_hybrid_fallback_on_empty(self, store):
        """Empty store should not crash with hybrid search."""
        query_emb = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)
        results = store.search(query_emb, limit=5, query_text="anything")
        assert results == []

    def test_backward_compat_no_query_text(self, store):
        """Without query_text, search behaves identically to pure vector."""
        texts = ["hello world", "foo bar baz"]
        chunks, embeddings = _make_chunks(texts)
        store.add_chunks(chunks, embeddings)

        query_emb = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)

        results_default = store.search(query_emb, limit=2)
        results_none = store.search(query_emb, limit=2, query_text=None)

        # Same results — both pure vector
        assert [cid for cid, _, _ in results_default] == [cid for cid, _, _ in results_none]

    def test_hybrid_with_domain_filter(self, store):
        """Domain filtering works with hybrid search."""
        chunks, embeddings = _make_chunks(["domain specific content about widgets"])
        store.add_chunks(chunks, embeddings, domain_id="domain_a")

        query_emb = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)

        # Search with matching domain
        results = store.search(query_emb, limit=5, domain_ids=["domain_a"], query_text="widgets")
        assert len(results) == 1

        # Search with non-matching domain — should find nothing
        results = store.search(query_emb, limit=5, domain_ids=["domain_b"], query_text="widgets")
        assert len(results) == 0

    def test_search_by_source_hybrid(self, store):
        """source-filtered hybrid search works."""
        texts_schema = ["users table with id, name, email columns"]
        texts_doc = ["documentation about user management"]
        chunks_s, emb_s = _make_chunks(texts_schema, source="schema")
        chunks_d, emb_d = _make_chunks(texts_doc, source="document")
        # Ensure unique doc names
        chunks_d[0].document_name = "doc_d_0"
        store.add_chunks(chunks_s, emb_s, source="schema")
        store.add_chunks(chunks_d, emb_d, source="document")

        query_emb = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)

        results = store.search_by_source(
            query_emb, source="schema", limit=5, min_similarity=0.0,
            query_text="users table",
        )
        # Should only return schema chunks
        for _, _, chunk in results:
            assert chunk.source == "schema"

    def test_hybrid_respects_limit(self, store):
        """Hybrid search respects the limit parameter."""
        texts = [f"document number {i} about topic {i}" for i in range(10)]
        chunks, embeddings = _make_chunks(texts)
        store.add_chunks(chunks, embeddings)

        query_emb = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)
        results = store.search(query_emb, limit=3, query_text="document topic")
        assert len(results) <= 3


class TestClustering:
    """Tests for glossary clustering."""

    def test_clusters_dirty_on_init(self, store):
        assert store._clusters_dirty is True

    def test_clusters_rebuild(self, store):
        """After rebuild, flag is False and clusters table is populated."""
        from constat.discovery.models import GlossaryTerm

        session_id = "test_session"
        # Add two glossary terms with embeddings
        chunks = [
            DocumentChunk(
                document_name="glossary:term_a",
                content="Customer: A person who buys",
                section="glossary",
                chunk_index=0,
                source="document",
                chunk_type=ChunkType.GLOSSARY_TERM,
            ),
            DocumentChunk(
                document_name="glossary:term_b",
                content="Order: A purchase transaction",
                section="glossary",
                chunk_index=0,
                source="document",
                chunk_type=ChunkType.GLOSSARY_TERM,
            ),
        ]
        embeddings = np.random.randn(2, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)
        store.add_chunks(chunks, embeddings, session_id=session_id)

        store._rebuild_clusters(session_id)
        assert store._clusters_dirty is False

        rows = store._conn.execute(
            "SELECT COUNT(*) FROM glossary_clusters WHERE session_id = ?",
            [session_id],
        ).fetchone()[0]
        assert rows == 2

    def test_clusters_dirty_on_glossary_add(self, store):
        from constat.discovery.models import GlossaryTerm

        # Force clean
        store._clusters_dirty = False
        term = GlossaryTerm(
            id="t1", name="customer", display_name="Customer",
            definition="A buyer", session_id="s1",
        )
        store.add_glossary_term(term)
        assert store._clusters_dirty is True

    def test_cluster_siblings(self, store):
        """Siblings query returns correct cluster members."""
        session_id = "test_session"
        # Create 4 chunks — with deterministic embeddings so clustering is predictable
        # Two similar pairs: (a, b) and (c, d)
        base_a = np.ones(DuckDBVectorStore.EMBEDDING_DIM, dtype=np.float32)
        base_b = np.ones(DuckDBVectorStore.EMBEDDING_DIM, dtype=np.float32) * 0.99
        base_c = -np.ones(DuckDBVectorStore.EMBEDDING_DIM, dtype=np.float32)
        base_d = -np.ones(DuckDBVectorStore.EMBEDDING_DIM, dtype=np.float32) * 0.99

        chunks = [
            DocumentChunk(document_name="glossary:a", content="A", section="glossary",
                          chunk_index=0, source="document", chunk_type=ChunkType.GLOSSARY_TERM),
            DocumentChunk(document_name="glossary:b", content="B", section="glossary",
                          chunk_index=0, source="document", chunk_type=ChunkType.GLOSSARY_TERM),
            DocumentChunk(document_name="glossary:c", content="C", section="glossary",
                          chunk_index=0, source="document", chunk_type=ChunkType.GLOSSARY_TERM),
            DocumentChunk(document_name="glossary:d", content="D", section="glossary",
                          chunk_index=0, source="document", chunk_type=ChunkType.GLOSSARY_TERM),
        ]
        embeddings = np.vstack([base_a, base_b, base_c, base_d])
        store.add_chunks(chunks, embeddings, session_id=session_id)

        siblings = store.get_cluster_siblings("a", session_id)
        # "a" and "b" should be in the same cluster
        assert "b" in siblings
        # "c" and "d" should NOT be siblings of "a"
        assert "c" not in siblings
        assert "d" not in siblings


class TestBM25Search:
    """Tests for BM25 search in isolation."""

    def test_bm25_keyword_match(self, store):
        """BM25 should find exact keyword matches."""
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A completely unrelated text about quantum physics",
        ]
        chunks, embeddings = _make_chunks(texts)
        store.add_chunks(chunks, embeddings)

        results = store._bm25_search("quick brown fox", limit=2)
        assert len(results) > 0
        # First result should contain "quick brown fox"
        assert "quick brown fox" in results[0][2].content

    def test_bm25_empty_store(self, store):
        """BM25 on empty store returns empty list."""
        results = store._bm25_search("anything", limit=5)
        assert results == []

    def test_bm25_with_source_filter(self, store):
        """BM25 respects source filter."""
        chunks_s, emb_s = _make_chunks(["schema table users"], source="schema")
        chunks_d, emb_d = _make_chunks(["document about users"], source="document")
        chunks_d[0].document_name = "doc_d_0"
        store.add_chunks(chunks_s, emb_s, source="schema")
        store.add_chunks(chunks_d, emb_d, source="document")

        results = store._bm25_search(
            "users", limit=5,
            source_filter="source = ?", source_params=["schema"],
        )
        for _, _, chunk in results:
            assert chunk.source == "schema"


class TestReranker:
    """Tests for cross-encoder reranking."""

    @pytest.fixture
    def reranker_store(self, tmp_path):
        """Create a DuckDBVectorStore with reranker enabled (mocked loader)."""
        db_path = str(tmp_path / "test_rerank.duckdb")
        os.environ["CONSTAT_VECTOR_STORE_PATH"] = db_path
        with patch("constat.reranker_loader.RerankerModelLoader") as mock_loader_cls:
            mock_instance = MagicMock()
            mock_loader_cls.get_instance.return_value = mock_instance
            s = DuckDBVectorStore(db_path=db_path, reranker_model="cross-encoder/test-model")
            mock_instance.start_loading.assert_called_once_with("cross-encoder/test-model")
        yield s
        s._db.close()
        os.environ.pop("CONSTAT_VECTOR_STORE_PATH", None)

    def test_rerank_reorders_results(self, reranker_store):
        """Reranker reorders candidates by cross-encoder score."""
        texts = ["irrelevant noise text", "exact match for query", "somewhat related"]
        chunks, embeddings = _make_chunks(texts)
        reranker_store.add_chunks(chunks, embeddings)

        mock_model = MagicMock()
        # Return scores that reverse the original order: last chunk scores highest
        mock_model.predict.return_value = [0.1, 5.0, 2.0]

        with patch("constat.reranker_loader.RerankerModelLoader") as mock_cls:
            mock_cls.get_instance.return_value.get_model.return_value = mock_model
            results = reranker_store._rerank(
                "exact match for query",
                [("id_0", 0.9, chunks[0]), ("id_1", 0.5, chunks[1]), ("id_2", 0.3, chunks[2])],
                limit=3,
            )

        # id_1 had highest raw score (5.0), should be first
        assert results[0][0] == "id_1"
        assert results[1][0] == "id_2"
        assert results[2][0] == "id_0"

    def test_rerank_normalizes_scores(self, reranker_store):
        """All reranked scores should be in [0, 1] via sigmoid."""
        c = DocumentChunk(document_name="a", content="a", section="", chunk_index=0, chunk_type=ChunkType.DOCUMENT)
        candidates = [("id_0", 0.9, c)]

        mock_model = MagicMock()
        mock_model.predict.return_value = [-10.0]

        with patch("constat.reranker_loader.RerankerModelLoader") as mock_cls:
            mock_cls.get_instance.return_value.get_model.return_value = mock_model
            results = reranker_store._rerank("query", candidates, limit=1)

        for _, score, _ in results:
            assert 0.0 <= score <= 1.0

    def test_rerank_disabled_by_default(self, tmp_path):
        """No reranking when reranker_model is not configured."""
        db_path = str(tmp_path / "test_no_rerank.duckdb")
        os.environ["CONSTAT_VECTOR_STORE_PATH"] = db_path
        s = DuckDBVectorStore(db_path=db_path)
        assert s._reranker_model is None

        texts = ["hello world", "foo bar"]
        chunks, embeddings = _make_chunks(texts)
        s.add_chunks(chunks, embeddings)

        query_emb = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)
        # Should work without any reranker calls
        results = s.search(query_emb, limit=2, query_text="hello")
        assert len(results) <= 2
        s._db.close()
        os.environ.pop("CONSTAT_VECTOR_STORE_PATH", None)

    def test_rerank_respects_limit(self, reranker_store):
        """Reranker returns at most `limit` results."""
        chunks = []
        for i in range(5):
            chunks.append(DocumentChunk(
                document_name=f"doc_{i}", content=f"content {i}",
                section="", chunk_index=i, chunk_type=ChunkType.DOCUMENT,
            ))
        candidates = [(f"id_{i}", 0.5, c) for i, c in enumerate(chunks)]

        mock_model = MagicMock()
        mock_model.predict.return_value = [1.0, 2.0, 3.0, 4.0, 5.0]

        with patch("constat.reranker_loader.RerankerModelLoader") as mock_cls:
            mock_cls.get_instance.return_value.get_model.return_value = mock_model
            results = reranker_store._rerank("query", candidates, limit=2)

        assert len(results) == 2

    def test_rerank_skipped_without_query_text(self, reranker_store):
        """Search without query_text should not trigger reranking."""
        texts = ["hello world", "foo bar"]
        chunks, embeddings = _make_chunks(texts)
        reranker_store.add_chunks(chunks, embeddings)

        query_emb = np.random.randn(1, DuckDBVectorStore.EMBEDDING_DIM).astype(np.float32)

        with patch("constat.reranker_loader.RerankerModelLoader") as mock_cls:
            results = reranker_store.search(query_emb, limit=2)
            # get_model should not be called when query_text is None
            mock_cls.get_instance.return_value.get_model.assert_not_called()
