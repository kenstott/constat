# Copyright (c) 2025 Kenneth Stott
# Canary: 3cac9ca0-8db8-4ea1-a735-20e498c907f1
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Abstract base class for vector storage backends.

Defines the full vector operations contract used by DuckDBVectorBackend.
"""

from abc import ABC, abstractmethod

import numpy as np

from constat.discovery.models import DocumentChunk


class VectorBackend(ABC):
    """ABC covering all vector/embedding operations on the embeddings table."""

    EMBEDDING_DIM = 1024

    @abstractmethod
    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        source: str = "document",
        session_id: str | None = None,
        domain_id: str | None = None,
    ) -> None:
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        domain_ids: list[str] | None = None,
        session_id: str | None = None,
        chunk_types: list[str] | None = None,
        query_text: str | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]:
        ...

    @abstractmethod
    def search_by_source(
        self,
        query_embedding: np.ndarray,
        source: str,
        limit: int = 5,
        min_similarity: float = 0.3,
        query_text: str | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]:
        ...

    @abstractmethod
    def delete_by_document(self, document_name: str) -> int:
        ...

    @abstractmethod
    def delete_by_source(self, source: str, domain_id: str | None = None) -> int:
        ...

    @abstractmethod
    def get_all_chunk_ids(self, session_id: str | None = None, global_only: bool = False) -> list[str]:
        ...

    @abstractmethod
    def get_chunks(self) -> list[DocumentChunk]:
        ...

    @abstractmethod
    def get_all_chunks(self, domain_ids: list[str] | None = None) -> list[DocumentChunk]:
        ...

    @abstractmethod
    def get_domain_chunks(self, domain_id: str) -> list[DocumentChunk]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def clear_chunks(self, source: str) -> None:
        ...

    @abstractmethod
    def clear_domain_embeddings(self, domain_id: str) -> int:
        ...

    @abstractmethod
    def count(self, source: str | None = None) -> int:
        ...

    @abstractmethod
    def rebuild_fts_index(self) -> None:
        ...

    @abstractmethod
    def store_document_url(self, document_name: str, source_url: str) -> None:
        ...

    @abstractmethod
    def get_document_url(self, document_name: str) -> str | None:
        ...

    @staticmethod
    @abstractmethod
    def chunk_visibility_filter(
        domain_ids: list[str] | None = None,
        alias: str = "",
    ) -> tuple[str, list]:
        ...

    @staticmethod
    @abstractmethod
    def _rows_to_chunks(rows: list) -> list[DocumentChunk]:
        ...
