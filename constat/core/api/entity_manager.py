# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Entity management for sessions.

Handles entity extraction lifecycle events:
- New session: Extract entities for base + active domains
- Domain add: Incrementally add entities for domain's chunks
- Domain delete: Incrementally delete entities for that domain
- Source add: Incrementally add entities for new source's chunks
- Source delete: Incrementally delete entities for that source

This module centralizes entity business logic that was previously
scattered across routes and session management code.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class VectorStoreProtocol(Protocol):
    """Protocol for vector store operations needed by EntityManager."""

    def clear_session_entities(self, session_id: str) -> None:
        """Clear all entities for a session."""
        ...

    def clear_domain_session_entities(self, session_id: str, domain_id: str) -> int:
        """Clear entities for a specific domain in a session."""
        ...

    def extract_entities_for_session(
        self,
        session_id: str,
        domain_ids: list[str] | None = None,
        schema_terms: list[str] | None = None,
        api_terms: list[str] | None = None,
        business_terms: list[str] | None = None,
    ) -> int:
        """Extract entities for all visible chunks in a session."""
        ...

    def extract_entities_for_domain(
        self,
        session_id: str,
        domain_id: str,
        schema_terms: list[str] | None = None,
        api_terms: list[str] | None = None,
        business_terms: list[str] | None = None,
    ) -> int:
        """Extract entities for a specific domain's chunks."""
        ...


@dataclass
class EntityExtractionResult:
    """Result of an entity extraction operation."""

    session_id: str
    entities_added: int = 0
    entities_removed: int = 0
    domains_processed: list[str] | None = None
    error: str | None = None


class EntityManager:
    """Manages entity extraction lifecycle for sessions.

    Provides a clean API for entity management operations, decoupling
    the business logic from routes and session management.
    """

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        schema_terms_provider: Optional[callable] = None,
        api_terms_provider: Optional[callable] = None,
    ):
        """Initialize the entity manager.

        Args:
            vector_store: Vector store backend for entity storage
            schema_terms_provider: Callable returning list of schema terms
            api_terms_provider: Callable returning list of API terms
        """
        self._vector_store = vector_store
        self._schema_terms_provider = schema_terms_provider
        self._api_terms_provider = api_terms_provider

    def _get_schema_terms(self) -> list[str]:
        """Get current schema terms for NER patterns."""
        if self._schema_terms_provider:
            return list(self._schema_terms_provider())
        return []

    def _get_api_terms(self) -> list[str]:
        """Get current API terms for NER patterns."""
        if self._api_terms_provider:
            return list(self._api_terms_provider())
        return []

    def extract_for_session(
        self,
        session_id: str,
        domain_ids: list[str] | None = None,
    ) -> EntityExtractionResult:
        """Extract entities for a new session or full refresh.

        Clears existing entities and re-extracts from all visible chunks
        (base config + active domains).

        Args:
            session_id: Session ID
            domain_ids: Active domain IDs to include

        Returns:
            EntityExtractionResult with extraction stats
        """
        try:
            count = self._vector_store.extract_entities_for_session(
                session_id=session_id,
                domain_ids=domain_ids,
                schema_terms=self._get_schema_terms(),
                api_terms=self._get_api_terms(),
            )
            logger.info(f"Session {session_id}: extracted {count} entities")
            return EntityExtractionResult(
                session_id=session_id,
                entities_added=count,
                domains_processed=domain_ids,
            )
        except Exception as e:
            logger.exception(f"Error extracting entities for session {session_id}")
            return EntityExtractionResult(
                session_id=session_id,
                error=str(e),
            )

    def add_domain(
        self,
        session_id: str,
        domain_id: str,
    ) -> EntityExtractionResult:
        """Incrementally add entities for a newly activated domain.

        Extracts entities from the domain's chunks without clearing
        existing session entities.

        Args:
            session_id: Session ID
            domain_id: Domain ID being activated

        Returns:
            EntityExtractionResult with extraction stats
        """
        try:
            count = self._vector_store.extract_entities_for_domain(
                session_id=session_id,
                domain_id=domain_id,
                schema_terms=self._get_schema_terms(),
                api_terms=self._get_api_terms(),
            )
            logger.info(f"Session {session_id}: added {count} entities for domain {domain_id}")
            return EntityExtractionResult(
                session_id=session_id,
                entities_added=count,
                domains_processed=[domain_id],
            )
        except Exception as e:
            logger.exception(f"Error adding entities for domain {domain_id}")
            return EntityExtractionResult(
                session_id=session_id,
                error=str(e),
            )

    def remove_domain(
        self,
        session_id: str,
        domain_id: str,
    ) -> EntityExtractionResult:
        """Incrementally remove entities for a deactivated domain.

        Clears entities associated with the domain's chunks.

        Args:
            session_id: Session ID
            domain_id: Domain ID being deactivated

        Returns:
            EntityExtractionResult with removal stats
        """
        try:
            count = self._vector_store.clear_domain_session_entities(
                session_id=session_id,
                domain_id=domain_id,
            )
            logger.info(f"Session {session_id}: removed {count} entities for domain {domain_id}")
            return EntityExtractionResult(
                session_id=session_id,
                entities_removed=count,
                domains_processed=[domain_id],
            )
        except Exception as e:
            logger.exception(f"Error removing entities for domain {domain_id}")
            return EntityExtractionResult(
                session_id=session_id,
                error=str(e),
            )

    def update_domains(
        self,
        session_id: str,
        old_domains: set[str],
        new_domains: set[str],
    ) -> EntityExtractionResult:
        """Handle domain changes - add/remove entities incrementally.

        Computes the diff between old and new domain sets and
        incrementally updates entities.

        Args:
            session_id: Session ID
            old_domains: Previously active domain IDs
            new_domains: Newly active domain IDs

        Returns:
            EntityExtractionResult with combined stats
        """
        removed_domains = old_domains - new_domains
        added_domains = new_domains - old_domains

        total_removed = 0
        total_added = 0
        errors = []

        # Remove entities for deactivated domains
        for domain_id in removed_domains:
            result = self.remove_domain(session_id, domain_id)
            if result.error:
                errors.append(f"{domain_id}: {result.error}")
            else:
                total_removed += result.entities_removed

        # Add entities for newly activated domains
        for domain_id in added_domains:
            result = self.add_domain(session_id, domain_id)
            if result.error:
                errors.append(f"{domain_id}: {result.error}")
            else:
                total_added += result.entities_added

        return EntityExtractionResult(
            session_id=session_id,
            entities_added=total_added,
            entities_removed=total_removed,
            domains_processed=list(removed_domains | added_domains),
            error="; ".join(errors) if errors else None,
        )

    def clear_session(self, session_id: str) -> None:
        """Clear all entities for a session.

        Args:
            session_id: Session ID to clear
        """
        self._vector_store.clear_session_entities(session_id)
        logger.info(f"Session {session_id}: cleared all entities")
