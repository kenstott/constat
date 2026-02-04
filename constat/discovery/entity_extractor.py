# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Entity extraction for document chunks using spaCy NER.

Extracts entities during session startup to enable entity-aware search
and cross-resource linking (documents <-> tables <-> APIs).

Uses a single spaCy pipeline with:
1. Built-in NER for generic entities (ORG, PERSON, LOCATION, etc.)
2. EntityRuler for custom patterns (schema terms, API terms, business terms)
"""

import hashlib
import logging
import re
from typing import Optional

import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler

from constat.discovery.models import (
    DocumentChunk,
    Entity,
    ChunkEntity,
    SemanticType,
    NerType,
    normalize_entity_name,
    display_entity_name,
)

logger = logging.getLogger(__name__)

# Load spaCy model once at module level
_nlp: Optional[Language] = None


def get_nlp() -> Language:
    """Get or load the spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


class EntityExtractor:
    """Extracts entities from document chunks using spaCy NER.

    Uses a single spaCy pipeline with custom EntityRuler patterns for
    schema terms, API terms, and business terms. This allows all entity
    extraction in one pass.

    Entity types:
    - Built-in NER: ORG, PERSON, GPE, PRODUCT, EVENT, etc.
    - Custom: SCHEMA (tables/columns), API (endpoints), TERM (business terms)
    """

    # spaCy entity types to keep from built-in NER
    KEEP_SPACY_TYPES = {'ORG', 'PRODUCT', 'GPE', 'EVENT', 'LAW', 'PERSON'}

    # Noise patterns to filter out
    NOISE_PATTERN = re.compile(r'^[\s#*_\-=`~\[\](){}|\\/<>@!$%^&+]+$')

    # Common noise words
    NOISE_WORDS = {
        'level', 'type', 'status', 'category', 'name', 'email', 'phone',
        'address', 'date', 'time', 'id', 'page', 'section', 'table',
    }

    def __init__(
        self,
        session_id: str,
        project_id: Optional[str] = None,
        schema_terms: Optional[list[str]] = None,
        api_terms: Optional[list[str]] = None,
        business_terms: Optional[list[str]] = None,
    ):
        """Initialize the entity extractor.

        Args:
            session_id: Session ID (required - entities are session-scoped)
            project_id: Optional project ID
            schema_terms: Database table/column names to recognize
            api_terms: API endpoint names to recognize
            business_terms: Business glossary terms to recognize
        """
        self.session_id = session_id
        self.project_id = project_id

        # Build custom NLP pipeline with EntityRuler
        self._nlp = get_nlp()

        # Add EntityRuler for custom patterns (before NER to take precedence)
        if "custom_ruler" in self._nlp.pipe_names:
            self._nlp.remove_pipe("custom_ruler")

        ruler = self._nlp.add_pipe("entity_ruler", name="custom_ruler", before="ner")

        patterns = []

        # Add schema patterns (multiple case variants for matching)
        for term in (schema_terms or []):
            if term and len(term) > 1:
                # Add original form
                patterns.append({"label": "SCHEMA", "pattern": term})
                # Add normalized (lowercase) form
                normalized = normalize_entity_name(term, to_singular=False)
                if normalized != term.lower():
                    patterns.append({"label": "SCHEMA", "pattern": normalized})
                # Add title case form for matching in prose
                title_case = normalized.title()
                if title_case != normalized and title_case != term:
                    patterns.append({"label": "SCHEMA", "pattern": title_case})
                # Add lowercase form
                lowercase = normalized.lower()
                if lowercase != normalized:
                    patterns.append({"label": "SCHEMA", "pattern": lowercase})

        # Add API patterns
        for term in (api_terms or []):
            if term and len(term) > 1:
                patterns.append({"label": "API", "pattern": term})

        # Add business term patterns
        for term in (business_terms or []):
            if term and len(term) > 1:
                patterns.append({"label": "TERM", "pattern": term})

        if patterns:
            ruler.add_patterns(patterns)
            logger.debug(f"EntityExtractor: added {len(patterns)} custom patterns")

        # Entity cache for deduplication
        self._entity_cache: dict[str, Entity] = {}

    def _generate_entity_id(self, name: str) -> str:
        """Generate entity ID from normalized name + session."""
        normalized = normalize_entity_name(name).lower()
        return hashlib.sha256(f"{normalized}:{self.session_id}".encode()).hexdigest()[:16]

    def _is_noise(self, text: str) -> bool:
        """Check if text is noise (symbols, too short, etc.)."""
        if len(text) < 2:
            return True
        if self.NOISE_PATTERN.search(text):
            return True
        if text.lower() in self.NOISE_WORDS:
            return True
        # Skip if mostly non-alphanumeric
        alpha_count = sum(1 for c in text if c.isalnum())
        if alpha_count < len(text) * 0.5:
            return True
        return False

    def _get_or_create_entity(self, name: str, spacy_label: str) -> Entity:
        """Get cached entity or create new one.

        Args:
            name: Raw entity name from text
            spacy_label: Entity label from spaCy (e.g., 'ORG', 'SCHEMA', 'API')

        Returns:
            Entity with semantic_type and ner_type set appropriately
        """
        normalized = normalize_entity_name(name).lower()

        if normalized in self._entity_cache:
            return self._entity_cache[normalized]

        # Determine semantic type based on source
        semantic_type = self._label_to_semantic_type(spacy_label)

        # Determine NER type (None for custom patterns)
        ner_type = spacy_label if spacy_label in self.KEEP_SPACY_TYPES else None

        entity = Entity(
            id=self._generate_entity_id(name),
            name=normalized,  # Normalized for NER matching
            display_name=display_entity_name(name),  # Title case for display
            semantic_type=semantic_type,
            ner_type=ner_type,
            session_id=self.session_id,
            project_id=self.project_id,
        )
        self._entity_cache[normalized] = entity
        return entity

    def _label_to_semantic_type(self, label: str) -> str:
        """Map spaCy/custom label to semantic type.

        Args:
            label: Entity label (e.g., 'ORG', 'SCHEMA', 'API', 'TERM')

        Returns:
            SemanticType value
        """
        # Custom pattern labels
        if label == 'SCHEMA':
            return SemanticType.CONCEPT  # Tables, columns are things
        elif label == 'API':
            return SemanticType.ACTION  # Endpoints are operations
        elif label == 'TERM':
            return SemanticType.TERM  # Business glossary terms

        # spaCy NER labels - most are concepts (nouns/things)
        if label in {'ORG', 'PERSON', 'PRODUCT', 'GPE', 'LAW'}:
            return SemanticType.CONCEPT
        elif label == 'EVENT':
            return SemanticType.ACTION  # Events can be actions

        return SemanticType.CONCEPT  # Default to concept

    def extract(self, chunk: DocumentChunk) -> list[tuple[Entity, ChunkEntity]]:
        """Extract entities from a chunk using spaCy NER.

        Args:
            chunk: Document chunk to extract entities from

        Returns:
            List of (Entity, ChunkEntity) tuples
        """
        doc = self._nlp(chunk.content)
        chunk_id = self._generate_chunk_id(chunk)

        results: list[tuple[Entity, ChunkEntity]] = []
        seen_entities: set[str] = set()

        for ent in doc.ents:
            text = ent.text.strip()

            # Skip noise
            if self._is_noise(text):
                continue

            # Only keep relevant entity types
            if ent.label_ not in self.KEEP_SPACY_TYPES and ent.label_ not in {'SCHEMA', 'API', 'TERM'}:
                continue

            # Deduplicate within chunk
            normalized = normalize_entity_name(text).lower()
            if normalized in seen_entities:
                continue
            seen_entities.add(normalized)

            entity = self._get_or_create_entity(text, ent.label_)

            link = ChunkEntity(
                chunk_id=chunk_id,
                entity_id=entity.id,
                confidence=0.9 if ent.label_ in {'SCHEMA', 'API', 'TERM'} else 0.75,
            )

            results.append((entity, link))

        return results

    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        """Generate chunk ID matching vector store format."""
        content_hash = hashlib.sha256(
            f"{chunk.document_name}:{chunk.section}:{chunk.chunk_index}:{chunk.content[:100]}".encode()
        ).hexdigest()[:16]
        return f"{chunk.document_name}_{chunk.chunk_index}_{content_hash}"

    def get_all_entities(self) -> list[Entity]:
        """Get all extracted entities (for batch insert)."""
        return list(self._entity_cache.values())

    def clear_cache(self) -> None:
        """Clear the entity cache."""
        self._entity_cache.clear()
