"""Entity extraction for document chunks.

Extracts entities during document ingestion to enable entity-aware search
and the explore_entity LLM tool.

Extraction sources:
1. Schema entities - Tables, columns from database metadata (high confidence)
2. Named entities - NER for business terms, proper nouns (medium confidence)
3. Domain concepts - LLM-assisted extraction for domain-specific terms
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional, Callable

from constat.discovery.models import (
    DocumentChunk,
    Entity,
    EntityType,
    ChunkEntity,
)


@dataclass
class ExtractionConfig:
    """Configuration for entity extraction."""

    # Whether to extract schema entities (tables, columns)
    extract_schema: bool = True

    # Whether to use NER for named entity extraction
    extract_ner: bool = True

    # Whether to use LLM for domain concept extraction
    extract_concepts: bool = False  # Off by default - requires LLM calls

    # Minimum confidence threshold for linking
    min_confidence: float = 0.5

    # Known schema entity names (tables, columns) for matching
    schema_entities: list[str] = field(default_factory=list)

    # Known business terms/glossary for matching
    business_terms: list[str] = field(default_factory=list)


class EntityExtractor:
    """Extracts entities from document chunks during ingestion.

    Supports three extraction approaches:
    1. Schema matching - Matches known table/column names in text
    2. Named entity recognition - Extracts proper nouns and business terms
    3. LLM-assisted - Uses LLM to identify domain concepts (optional)
    """

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        llm_extractor: Optional[Callable[[str], list[tuple[str, str]]]] = None,
    ):
        """Initialize the entity extractor.

        Args:
            config: Extraction configuration
            llm_extractor: Optional LLM function that takes text and returns
                          list of (entity_name, entity_type) tuples
        """
        self.config = config or ExtractionConfig()
        self._llm_extractor = llm_extractor

        # Build lookup sets for fast matching
        self._schema_patterns = self._build_patterns(self.config.schema_entities)
        self._term_patterns = self._build_patterns(self.config.business_terms)

        # Entity deduplication cache
        self._entity_cache: dict[str, Entity] = {}

    def _get_stem_variations(self, term: str) -> list[str]:
        """Generate common singular/plural variations of a term.

        Returns list of variations including the original term.
        """
        variations = [term]
        lower = term.lower()

        # Handle plurals -> singular
        if lower.endswith('ies') and len(lower) > 3:
            # categories -> category
            variations.append(term[:-3] + 'y')
        elif lower.endswith('es') and len(lower) > 2:
            # processes -> process, boxes -> box
            variations.append(term[:-2])
            # Also try just removing 's' for words like 'employees'
            variations.append(term[:-1])
        elif lower.endswith('s') and len(lower) > 1:
            # customers -> customer
            variations.append(term[:-1])

        # Handle singular -> plural
        if not lower.endswith('s'):
            if lower.endswith('y') and len(lower) > 1:
                # category -> categories
                variations.append(term[:-1] + 'ies')
            elif lower.endswith(('s', 'x', 'z', 'ch', 'sh')):
                # box -> boxes
                variations.append(term + 'es')
            else:
                # customer -> customers
                variations.append(term + 's')

        return variations

    def _build_patterns(self, terms: list[str]) -> dict[str, re.Pattern]:
        """Build regex patterns for term matching with stemming.

        Returns dict mapping lowercase term to compiled pattern.
        Patterns match whole words and common inflections, case-insensitive.
        """
        patterns = {}
        for term in terms:
            if not term:
                continue

            # Get all variations (singular/plural)
            variations = self._get_stem_variations(term)

            # Build alternation pattern for all variations
            escaped_variations = [re.escape(v) for v in variations]
            alternation = '|'.join(escaped_variations)
            pattern = re.compile(rf'\b({alternation})\b', re.IGNORECASE)
            patterns[term.lower()] = pattern
        return patterns

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate a stable entity ID from name and type."""
        key = f"{entity_type}:{name.lower()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _get_or_create_entity(
        self,
        name: str,
        entity_type: str,
        metadata: Optional[dict] = None,
    ) -> Entity:
        """Get cached entity or create new one.

        Ensures entity deduplication across chunks.
        """
        cache_key = f"{entity_type}:{name.lower()}"

        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        entity = Entity(
            id=self._generate_entity_id(name, entity_type),
            name=name,
            type=entity_type,
            metadata=metadata or {},
        )
        self._entity_cache[cache_key] = entity
        return entity

    def extract(self, chunk: DocumentChunk) -> list[tuple[Entity, ChunkEntity]]:
        """Extract entities from a chunk.

        Returns list of (Entity, ChunkEntity) tuples where:
        - Entity is the extracted entity (may be shared across chunks)
        - ChunkEntity is the link with mention count and confidence

        Args:
            chunk: Document chunk to extract entities from

        Returns:
            List of (Entity, ChunkEntity) tuples
        """
        text = chunk.content
        chunk_id = self._generate_chunk_id(chunk)

        results: list[tuple[Entity, ChunkEntity]] = []

        # 1. Schema entity extraction
        if self.config.extract_schema:
            results.extend(self._extract_schema_entities(text, chunk_id))

        # 2. Named entity recognition
        if self.config.extract_ner:
            results.extend(self._extract_named_entities(text, chunk_id))

        # 3. LLM-assisted concept extraction
        if self.config.extract_concepts and self._llm_extractor:
            results.extend(self._extract_concepts(text, chunk_id))

        return results

    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        """Generate chunk ID matching vector store format."""
        content_hash = hashlib.sha256(
            f"{chunk.document_name}:{chunk.section}:{chunk.chunk_index}:{chunk.content[:100]}".encode()
        ).hexdigest()[:16]
        return f"{chunk.document_name}_{chunk.chunk_index}_{content_hash}"

    def _extract_schema_entities(
        self,
        text: str,
        chunk_id: str,
    ) -> list[tuple[Entity, ChunkEntity]]:
        """Extract schema entities (tables, columns) by pattern matching."""
        results = []

        for term, pattern in self._schema_patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Use the first match's actual casing
                actual_name = matches[0]
                mention_count = len(matches)

                # Determine if it's a table or column based on naming conventions
                # Tables are usually PascalCase or snake_case nouns
                # Columns often have prefixes like id_, _at, etc.
                entity_type = EntityType.TABLE
                if '_id' in term or term.endswith('_at') or term.startswith('is_'):
                    entity_type = EntityType.COLUMN

                entity = self._get_or_create_entity(
                    name=actual_name,
                    entity_type=entity_type,
                    metadata={"source": "schema"},
                )

                link = ChunkEntity(
                    chunk_id=chunk_id,
                    entity_id=entity.id,
                    mention_count=mention_count,
                    confidence=0.95,  # High confidence for schema matches
                )

                results.append((entity, link))

        return results

    def _extract_named_entities(
        self,
        text: str,
        chunk_id: str,
    ) -> list[tuple[Entity, ChunkEntity]]:
        """Extract named entities using pattern-based NER.

        Extracts:
        - Known business terms from glossary
        - CamelCase/PascalCase identifiers (likely class/type names)
        - ALL_CAPS identifiers (likely constants or acronyms)
        """
        results = []

        # Match known business terms
        for term, pattern in self._term_patterns.items():
            matches = pattern.findall(text)
            if matches:
                actual_name = matches[0]
                mention_count = len(matches)

                entity = self._get_or_create_entity(
                    name=actual_name,
                    entity_type=EntityType.BUSINESS_TERM,
                    metadata={"source": "glossary"},
                )

                link = ChunkEntity(
                    chunk_id=chunk_id,
                    entity_id=entity.id,
                    mention_count=mention_count,
                    confidence=0.85,
                )

                results.append((entity, link))

        # Extract PascalCase identifiers (likely class/concept names)
        pascal_pattern = re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b')
        for match in pascal_pattern.finditer(text):
            name = match.group(1)
            # Skip if already matched as schema entity
            if name.lower() in self._schema_patterns:
                continue

            entity = self._get_or_create_entity(
                name=name,
                entity_type=EntityType.CONCEPT,
                metadata={"source": "ner", "pattern": "PascalCase"},
            )

            link = ChunkEntity(
                chunk_id=chunk_id,
                entity_id=entity.id,
                mention_count=1,
                confidence=0.7,
            )

            results.append((entity, link))

        # Extract ALL_CAPS identifiers (acronyms, constants)
        caps_pattern = re.compile(r'\b([A-Z][A-Z0-9_]{2,})\b')
        for match in caps_pattern.finditer(text):
            name = match.group(1)
            # Skip common words that happen to be caps
            if name in ('THE', 'AND', 'FOR', 'NOT', 'SQL', 'API', 'URL', 'JSON', 'XML'):
                continue

            entity = self._get_or_create_entity(
                name=name,
                entity_type=EntityType.BUSINESS_TERM,
                metadata={"source": "ner", "pattern": "CAPS"},
            )

            link = ChunkEntity(
                chunk_id=chunk_id,
                entity_id=entity.id,
                mention_count=1,
                confidence=0.6,
            )

            results.append((entity, link))

        return results

    def _extract_concepts(
        self,
        text: str,
        chunk_id: str,
    ) -> list[tuple[Entity, ChunkEntity]]:
        """Extract domain concepts using LLM.

        Requires llm_extractor function to be provided.
        """
        if not self._llm_extractor:
            return []

        try:
            extracted = self._llm_extractor(text)

            results = []
            for name, entity_type in extracted:
                # Map LLM-provided type to our types
                if entity_type.lower() in ('table', 'column'):
                    etype = entity_type.lower()
                elif entity_type.lower() in ('term', 'business_term', 'glossary'):
                    etype = EntityType.BUSINESS_TERM
                else:
                    etype = EntityType.CONCEPT

                entity = self._get_or_create_entity(
                    name=name,
                    entity_type=etype,
                    metadata={"source": "llm"},
                )

                link = ChunkEntity(
                    chunk_id=chunk_id,
                    entity_id=entity.id,
                    mention_count=1,
                    confidence=0.75,  # Medium confidence for LLM extraction
                )

                results.append((entity, link))

            return results

        except Exception:
            # LLM extraction failed, return empty
            return []

    def get_all_entities(self) -> list[Entity]:
        """Get all extracted entities (for batch insert)."""
        return list(self._entity_cache.values())

    def clear_cache(self) -> None:
        """Clear the entity cache."""
        self._entity_cache.clear()

    def update_schema_entities(self, entities: list[str]) -> None:
        """Update the list of known schema entities.

        Call this after database introspection to enable schema matching.
        """
        self.config.schema_entities = entities
        self._schema_patterns = self._build_patterns(entities)

    def update_business_terms(self, terms: list[str]) -> None:
        """Update the list of known business terms.

        Call this to add domain-specific vocabulary.
        """
        self.config.business_terms = terms
        self._term_patterns = self._build_patterns(terms)


def create_schema_entities_from_catalog(
    tables: list[str],
    columns: list[str],
) -> list[Entity]:
    """Create Entity objects from database catalog information.

    Args:
        tables: List of table names
        columns: List of column names

    Returns:
        List of Entity objects for tables and columns
    """
    entities = []

    for table in tables:
        entity_id = hashlib.sha256(f"table:{table.lower()}".encode()).hexdigest()[:16]
        entities.append(Entity(
            id=entity_id,
            name=table,
            type=EntityType.TABLE,
            metadata={"source": "catalog"},
        ))

    for column in columns:
        entity_id = hashlib.sha256(f"column:{column.lower()}".encode()).hexdigest()[:16]
        entities.append(Entity(
            id=entity_id,
            name=column,
            type=EntityType.COLUMN,
            metadata={"source": "catalog"},
        ))

    return entities