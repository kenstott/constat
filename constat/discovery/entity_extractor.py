# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Entity extraction for document chunks using spaCy NER.

Extracts entities during document ingestion to enable entity-aware search
and the explore_entity LLM tool.

Extraction sources:
1. Database schemas - Tables, columns from database metadata
2. OpenAPI schemas - Endpoints, operations, schema definitions
3. GraphQL schemas - Types, fields, queries, mutations
4. Document NER - Named entities using spaCy (ORG, PRODUCT, GPE, etc.)
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import spacy

logger = logging.getLogger(__name__)
from spacy.language import Language

from constat.discovery.models import (
    DocumentChunk,
    Entity,
    EntityType,
    ChunkEntity,
    normalize_entity_name,
    display_entity_name,
)

# Load spaCy model once at module level
_nlp: Optional[Language] = None


def get_nlp() -> Language:
    """Get or load the spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


@dataclass
class ExtractionConfig:
    """Configuration for entity extraction."""

    # Whether to extract schema entities (tables, columns)
    extract_schema: bool = True

    # Whether to use spaCy NER for named entity extraction
    extract_ner: bool = True

    # Minimum confidence threshold for linking
    min_confidence: float = 0.5

    # Known schema entity names (tables, columns) for matching
    schema_entities: list[str] = field(default_factory=list)

    # Known business terms/glossary for matching
    business_terms: list[str] = field(default_factory=list)

    # OpenAPI operations (endpoint names)
    openapi_operations: list[str] = field(default_factory=list)

    # OpenAPI schema definitions
    openapi_schemas: list[str] = field(default_factory=list)

    # GraphQL types
    graphql_types: list[str] = field(default_factory=list)

    # GraphQL fields/operations
    graphql_fields: list[str] = field(default_factory=list)


class EntityExtractor:
    """Extracts entities from document chunks during ingestion.

    Uses spaCy NER for document content and pattern matching for
    database, OpenAPI, and GraphQL schemas.
    """

    # spaCy entity types to extract
    # Excluding: MONEY, PERCENT, CARDINAL (numeric), PERSON (too many false positives)
    SPACY_ENTITY_TYPES = {
        'ORG',      # Organizations, companies
        'PRODUCT',  # Products, software
        'GPE',      # Countries, cities, states (geographic)
        'EVENT',    # Named events
        'LAW',      # Legal documents, treaties
    }

    # Patterns to filter out (markdown artifacts, symbols, etc.)
    NOISE_PATTERNS = re.compile(
        r'^[\s#*_\-=`~\[\](){}|\\/<>@!$%^&+]+$'  # Pure symbols/markdown
        r'|^#+\s*$'                               # Markdown headers
        r'|^\*+$'                                 # Markdown bold/italic
        r'|^-+$'                                  # Markdown hr
        r'|^_{2,}$'                               # Underscores
        r'|[+/]'                                  # Contains + or / (e.g., "Email + Chat")
    )

    # Common words that spaCy misclassifies as entities
    # Keep this minimal - only filter true noise, not business terms
    NOISE_WORDS = {
        # Generic structural terms
        'level', 'type', 'status', 'category',
        # Actions
        'create', 'update', 'delete', 'read', 'write', 'send', 'receive',
        # Common columns/fields that are too generic
        'name', 'email', 'phone', 'address', 'date', 'time', 'id',
        # Time periods
        'q1', 'q2', 'q3', 'q4', 'year', 'month', 'week', 'day', 'hour',
        'fy', 'ytd', 'mtd', 'qtd',
        # Document structure terms
        'page', 'section', 'table', 'figure', 'appendix', 'chapter',
    }

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
    ):
        """Initialize the entity extractor.

        Args:
            config: Extraction configuration
        """
        self.config = config or ExtractionConfig()

        # Build lookup sets for fast matching
        self._schema_patterns = self._build_patterns(self.config.schema_entities)
        self._term_patterns = self._build_patterns(self.config.business_terms)
        self._openapi_op_patterns = self._build_patterns(self.config.openapi_operations)
        self._openapi_schema_patterns = self._build_patterns(self.config.openapi_schemas)
        self._graphql_type_patterns = self._build_patterns(self.config.graphql_types)
        self._graphql_field_patterns = self._build_patterns(self.config.graphql_fields)

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
            variations.append(term[:-3] + 'y')
        elif lower.endswith('es') and len(lower) > 2:
            variations.append(term[:-2])
            variations.append(term[:-1])
        elif lower.endswith('s') and len(lower) > 1:
            variations.append(term[:-1])

        # Handle singular -> plural
        if not lower.endswith('s'):
            if lower.endswith('y') and len(lower) > 1:
                variations.append(term[:-1] + 'ies')
            elif lower.endswith(('s', 'x', 'z', 'ch', 'sh')):
                variations.append(term + 'es')
            else:
                variations.append(term + 's')

        return variations

    def _build_patterns(self, terms: list[str]) -> dict[str, re.Pattern]:
        """Build regex patterns for term matching with stemming and normalization.

        Returns dict mapping original term to compiled pattern.
        Patterns match:
        - Original term (e.g., "order_id")
        - Normalized term (e.g., "order id")
        - Singular/plural variations of both
        All matching is case-insensitive.
        """
        patterns = {}
        for term in terms:
            if not term:
                continue

            # Get variations of original term
            variations = set(self._get_stem_variations(term))

            # Also get variations of normalized term (underscore/hyphen -> space)
            normalized = normalize_entity_name(term)
            if normalized != term:
                variations.update(self._get_stem_variations(normalized))

            escaped_variations = [re.escape(v) for v in variations]
            alternation = '|'.join(escaped_variations)
            pattern = re.compile(rf'\b({alternation})\b', re.IGNORECASE)
            patterns[term] = pattern  # Keep original term as key
            logger.debug(f"Built pattern for '{term}': variations={variations}")
        return patterns

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate a stable entity ID from normalized name and type.

        Uses normalized name so "order_id" and "order id" map to the same entity.
        """
        normalized = normalize_entity_name(name).lower()
        key = f"{entity_type}:{normalized}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _get_or_create_entity(
        self,
        name: str,
        entity_type: str,
        metadata: Optional[dict] = None,
    ) -> Entity:
        """Get cached entity or create new one.

        Ensures entity deduplication across chunks using normalized names.
        Stores:
        - name: Display name (title case, singular) e.g., "Order Item"
        - metadata.normalized: Normalized form (lowercase, singular) e.g., "order item"
        - metadata.original_name: Original as found e.g., "order_items"
        """
        # Use normalized name for cache key to dedupe "order_id" and "order id"
        normalized = normalize_entity_name(name)  # lowercase singular
        display_name = display_entity_name(name)  # title case singular
        cache_key = f"{entity_type}:{normalized.lower()}"

        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        # Store names in metadata
        entity_metadata = metadata.copy() if metadata else {}
        entity_metadata["normalized"] = normalized
        if name != normalized and name != display_name:
            entity_metadata["original_name"] = name

        entity = Entity(
            id=self._generate_entity_id(name, entity_type),
            name=display_name,  # Use display name (title case, singular)
            type=entity_type,
            metadata=entity_metadata,
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

        # 1. Database schema entity extraction
        if self.config.extract_schema:
            results.extend(self._extract_schema_entities(text, chunk_id))

        # 2. OpenAPI entity extraction
        results.extend(self._extract_openapi_entities(text, chunk_id))

        # 3. GraphQL entity extraction
        results.extend(self._extract_graphql_entities(text, chunk_id))

        # 4. Business term extraction
        results.extend(self._extract_business_terms(text, chunk_id))

        # 5. spaCy NER extraction
        if self.config.extract_ner:
            results.extend(self._extract_spacy_entities(text, chunk_id))

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
        """Extract database schema entities (tables, columns) by pattern matching."""
        results = []

        for term, pattern in self._schema_patterns.items():
            matches = pattern.findall(text)
            if matches:
                actual_name = matches[0]
                mention_count = len(matches)
                logger.debug(f"Schema match: '{term}' -> '{actual_name}' ({mention_count}x) in chunk {chunk_id[:8]}")

                # Determine if it's a table or column based on naming conventions
                entity_type = EntityType.TABLE
                if '_id' in term or term.endswith('_at') or term.startswith('is_'):
                    entity_type = EntityType.COLUMN

                entity = self._get_or_create_entity(
                    name=actual_name,
                    entity_type=entity_type,
                    metadata={"source": "database_schema"},
                )

                link = ChunkEntity(
                    chunk_id=chunk_id,
                    entity_id=entity.id,
                    mention_count=mention_count,
                    confidence=0.95,
                    mention_text=actual_name,
                )

                results.append((entity, link))

        return results

    def _extract_openapi_entities(
        self,
        text: str,
        chunk_id: str,
    ) -> list[tuple[Entity, ChunkEntity]]:
        """Extract OpenAPI entities (operations, schemas)."""
        results = []

        # Extract operations (endpoints)
        for term, pattern in self._openapi_op_patterns.items():
            matches = pattern.findall(text)
            if matches:
                actual_name = matches[0]
                mention_count = len(matches)

                entity = self._get_or_create_entity(
                    name=actual_name,
                    entity_type=EntityType.API_ENDPOINT,
                    metadata={"source": "openapi"},
                )

                link = ChunkEntity(
                    chunk_id=chunk_id,
                    entity_id=entity.id,
                    mention_count=mention_count,
                    confidence=0.95,
                    mention_text=actual_name,
                )

                results.append((entity, link))

        # Extract schema definitions
        for term, pattern in self._openapi_schema_patterns.items():
            matches = pattern.findall(text)
            if matches:
                actual_name = matches[0]
                mention_count = len(matches)

                entity = self._get_or_create_entity(
                    name=actual_name,
                    entity_type=EntityType.API_SCHEMA,
                    metadata={"source": "openapi"},
                )

                link = ChunkEntity(
                    chunk_id=chunk_id,
                    entity_id=entity.id,
                    mention_count=mention_count,
                    confidence=0.90,
                    mention_text=actual_name,
                )

                results.append((entity, link))

        return results

    def _extract_graphql_entities(
        self,
        text: str,
        chunk_id: str,
    ) -> list[tuple[Entity, ChunkEntity]]:
        """Extract GraphQL entities (types, fields)."""
        results = []

        # Extract types
        for term, pattern in self._graphql_type_patterns.items():
            matches = pattern.findall(text)
            if matches:
                actual_name = matches[0]
                mention_count = len(matches)

                entity = self._get_or_create_entity(
                    name=actual_name,
                    entity_type=EntityType.GRAPHQL_TYPE,
                    metadata={"source": "graphql"},
                )

                link = ChunkEntity(
                    chunk_id=chunk_id,
                    entity_id=entity.id,
                    mention_count=mention_count,
                    confidence=0.95,
                )

                results.append((entity, link))

        # Extract fields/operations
        for term, pattern in self._graphql_field_patterns.items():
            matches = pattern.findall(text)
            if matches:
                actual_name = matches[0]
                mention_count = len(matches)

                entity = self._get_or_create_entity(
                    name=actual_name,
                    entity_type=EntityType.GRAPHQL_FIELD,
                    metadata={"source": "graphql"},
                )

                link = ChunkEntity(
                    chunk_id=chunk_id,
                    entity_id=entity.id,
                    mention_count=mention_count,
                    confidence=0.90,
                    mention_text=actual_name,
                )

                results.append((entity, link))

        return results

    def _extract_business_terms(
        self,
        text: str,
        chunk_id: str,
    ) -> list[tuple[Entity, ChunkEntity]]:
        """Extract known business terms from glossary."""
        results = []

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
                    mention_text=actual_name,
                )

                results.append((entity, link))

        return results

    def _extract_spacy_entities(
        self,
        text: str,
        chunk_id: str,
    ) -> list[tuple[Entity, ChunkEntity]]:
        """Extract named entities using spaCy NER."""
        results = []
        nlp = get_nlp()
        doc = nlp(text)

        # Count mentions per entity
        entity_mentions: dict[tuple[str, str], int] = {}
        for ent in doc.ents:
            if ent.label_ in self.SPACY_ENTITY_TYPES:
                # Skip noise (markdown, symbols, short strings)
                text = ent.text.strip()
                if len(text) < 2:
                    continue
                if self.NOISE_PATTERNS.search(text):
                    continue
                # Skip if mostly non-alphanumeric
                alpha_count = sum(1 for c in text if c.isalnum())
                if alpha_count < len(text) * 0.5:
                    continue
                # Skip common noise words
                if text.lower() in self.NOISE_WORDS:
                    continue
                # Skip if any word is a noise word (for multi-word entities)
                words = text.lower().split()
                if any(w in self.NOISE_WORDS for w in words):
                    continue

                key = (text, ent.label_)
                entity_mentions[key] = entity_mentions.get(key, 0) + 1

        # Create entities
        for (mention_text, spacy_type), mention_count in entity_mentions.items():
            # Map spaCy type to our entity types
            entity_type = self._map_spacy_type(spacy_type)

            # Skip if already matched by schema/API patterns (higher confidence)
            normalized = normalize_entity_name(mention_text)
            cache_key = f"{entity_type}:{normalized.lower()}"
            if cache_key in self._entity_cache:
                continue

            entity = self._get_or_create_entity(
                name=mention_text,
                entity_type=entity_type,
                metadata={"source": "spacy_ner", "spacy_type": spacy_type},
            )

            link = ChunkEntity(
                chunk_id=chunk_id,
                entity_id=entity.id,
                mention_count=mention_count,
                confidence=0.75,  # NER confidence
                mention_text=mention_text,  # Store exact text as found
            )

            results.append((entity, link))

        return results

    def _map_spacy_type(self, spacy_type: str) -> str:
        """Map spaCy entity type to our entity type."""
        mapping = {
            'ORG': EntityType.ORGANIZATION,
            'PRODUCT': EntityType.PRODUCT,
            'GPE': EntityType.LOCATION,
            'PERSON': EntityType.PERSON,
            'EVENT': EntityType.EVENT,
            'WORK_OF_ART': EntityType.CONCEPT,
            'LAW': EntityType.CONCEPT,
            'MONEY': EntityType.METRIC,
            'PERCENT': EntityType.METRIC,
        }
        return mapping.get(spacy_type, EntityType.CONCEPT)

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

    def update_openapi_entities(
        self,
        operations: list[str],
        schemas: list[str],
    ) -> None:
        """Update OpenAPI entities for pattern matching.

        Args:
            operations: List of operation/endpoint names
            schemas: List of schema definition names
        """
        self.config.openapi_operations = operations
        self.config.openapi_schemas = schemas
        self._openapi_op_patterns = self._build_patterns(operations)
        self._openapi_schema_patterns = self._build_patterns(schemas)

    def update_graphql_entities(
        self,
        types: list[str],
        fields: list[str],
    ) -> None:
        """Update GraphQL entities for pattern matching.

        Args:
            types: List of type names
            fields: List of field/operation names
        """
        self.config.graphql_types = types
        self.config.graphql_fields = fields
        self._graphql_type_patterns = self._build_patterns(types)
        self._graphql_field_patterns = self._build_patterns(fields)


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
        normalized = normalize_entity_name(table)
        display_name = display_entity_name(table)
        entity_id = hashlib.sha256(f"table:{normalized.lower()}".encode()).hexdigest()[:16]
        entities.append(Entity(
            id=entity_id,
            name=display_name,
            type=EntityType.TABLE,
            metadata={
                "source": "catalog",
                "normalized": normalized,
                "original_name": table if table != normalized else None,
            },
        ))

    for column in columns:
        normalized = normalize_entity_name(column)
        display_name = display_entity_name(column)
        entity_id = hashlib.sha256(f"column:{normalized.lower()}".encode()).hexdigest()[:16]
        entities.append(Entity(
            id=entity_id,
            name=display_name,
            type=EntityType.COLUMN,
            metadata={
                "source": "catalog",
                "normalized": normalized,
                "original_name": column if column != normalized else None,
            },
        ))

    return entities
