# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Data models for document discovery."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from constat.core.config import DocumentConfig


class ChunkType(str, Enum):
    """Granular type for document chunks.

    Identifies what kind of resource a chunk describes.
    """
    # Database schema
    DB_TABLE = "db_table"
    DB_COLUMN = "db_column"
    DB_SCHEMA = "db_schema"

    # REST API
    API_ENDPOINT = "api_endpoint"
    API_SCHEMA = "api_schema"

    # GraphQL
    GRAPHQL_QUERY = "graphql_query"
    GRAPHQL_MUTATION = "graphql_mutation"
    GRAPHQL_TYPE = "graphql_type"
    GRAPHQL_FIELD = "graphql_field"

    # Documents
    DOCUMENT = "document"

    # Glossary & Relationships
    GLOSSARY_TERM = "glossary_term"
    RELATIONSHIP = "relationship"


def singularize(word: str) -> str:
    """Convert a word to its singular form.

    Simple heuristic-based singularization for common English patterns.

    Args:
        word: Word to singularize (e.g., "orders", "employees", "categories")

    Returns:
        Singular form (e.g., "order", "employee", "category")
    """
    if not word or len(word) < 3:
        return word

    lower = word.lower()

    # Handle common irregular plurals
    irregulars = {
        'people': 'person',
        'children': 'child',
        'men': 'man',
        'women': 'woman',
        'mice': 'mouse',
        'geese': 'goose',
        'teeth': 'tooth',
        'feet': 'foot',
        'data': 'datum',
        'criteria': 'criterion',
        'indices': 'index',
        'vertices': 'vertex',
        'matrices': 'matrix',
    }
    if lower in irregulars:
        # Preserve original case pattern
        if word[0].isupper():
            return irregulars[lower].capitalize()
        return irregulars[lower]

    # Don't singularize words ending in 'ss', 'us', 'is'
    if lower.endswith(('ss', 'us', 'is')):
        return word

    # Handle -ies -> -y (categories -> category)
    if lower.endswith('ies') and len(word) > 3:
        return word[:-3] + 'y'

    # Handle -es -> remove (boxes -> box, watches -> watch)
    if lower.endswith('es') and len(word) > 2:
        # Check for -ches, -shes, -xes, -zes, -ses
        if lower.endswith(('ches', 'shes', 'xes', 'zes', 'ses')):
            return word[:-2]
        # Check for -oes (heroes -> hero, but not shoes)
        if lower.endswith('oes') and lower not in ('shoes', 'toes'):
            return word[:-2]
        # Default: try removing -es, fall back to -s
        if lower.endswith('ves'):
            return word[:-3] + 'f'  # leaves -> leaf
        return word[:-1]  # Just remove the 's'

    # Handle simple -s -> remove
    if lower.endswith('s') and not lower.endswith('ss'):
        return word[:-1]

    return word


def is_camel_case(name: str) -> bool:
    """Detect if a name is in CamelCase or camelCase format.

    CamelCase names are schema/code identifiers that should be split into words.
    Examples: BillingAddress, firstName, OrderItems, getUserById, XMLParser

    Args:
        name: Name to check

    Returns:
        True if the name appears to be CamelCase
    """
    if not name or len(name) < 2:
        return False

    # Check for lowercase followed by uppercase (e.g., "firstName", "billingAddress")
    for i in range(len(name) - 1):
        if name[i].islower() and name[i + 1].isupper():
            return True

    # Check for uppercase sequence followed by uppercase+lowercase (e.g., "XMLParser")
    # Pattern: 2+ uppercase letters followed by uppercase+lowercase
    for i in range(len(name) - 2):
        if name[i].isupper() and name[i + 1].isupper() and name[i + 2].islower():
            return True

    return False


def split_camel_case(name: str) -> str:
    """Split CamelCase into separate words.

    Args:
        name: CamelCase name (e.g., "BillingAddress", "firstName")

    Returns:
        Space-separated words (e.g., "Billing Address", "first Name")
    """
    # Insert space before uppercase letters that follow lowercase letters
    result = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    # Also handle sequences like "XMLParser" -> "XML Parser"
    result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', result)
    return result


def is_proper_noun(name: str) -> bool:
    """Detect if a name appears to be a proper noun.

    Proper nouns (e.g., "Microsoft", "San Francisco") have intentional
    capitalization that should be preserved. Schema names (e.g.,
    "order_items", "user-profiles", "BillingAddress") should be normalized.

    Args:
        name: Entity name to check

    Returns:
        True if the name appears to be a proper noun
    """
    # If it contains underscores or hyphens, it's a schema name
    if '_' in name or '-' in name:
        return False

    # If it's all lowercase, it's not a proper noun
    if name == name.lower():
        return False

    # If it's CamelCase (e.g., "BillingAddress", "firstName"), it's a schema name
    if is_camel_case(name):
        return False

    # If it's all uppercase, treat as acronym (proper noun)
    if name == name.upper() and len(name) > 1:
        return True

    # Single capitalized word without CamelCase pattern (e.g., "Microsoft")
    # is likely a proper noun
    return True


def normalize_entity_name(name: str, to_singular: bool = True) -> str:
    """Normalize an entity name to a canonical form.

    Converts underscores, hyphens, and CamelCase to spaces, optionally singularizes.
    Preserves case for proper nouns, lowercases schema names.

    Args:
        name: Raw entity name (e.g., "order_items", "BillingAddress", "Microsoft")
        to_singular: If True, convert to singular form (default True)

    Returns:
        Normalized name (e.g., "order item", "billing address", "Microsoft")
    """
    proper_noun = is_proper_noun(name)

    # Split CamelCase first (before other transformations)
    if is_camel_case(name):
        normalized = split_camel_case(name)
    else:
        normalized = name

    # Replace underscores and hyphens with spaces
    normalized = re.sub(r'[_-]+', ' ', normalized)
    # Collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Singularize each word if requested
    if to_singular and normalized:
        words = normalized.split()
        words = [singularize(w) for w in words]
        normalized = ' '.join(words)

    # Lowercase non-proper nouns
    if not proper_noun:
        normalized = normalized.lower()

    return normalized


def display_entity_name(name: str) -> str:
    """Get the display name for an entity.

    For proper nouns (e.g., "Microsoft"), preserves original capitalization.
    For schema names (e.g., "order_items"), uses title case.

    Args:
        name: Raw or normalized entity name

    Returns:
        Display name (e.g., "Order Item" or "Microsoft")
    """
    proper_noun = is_proper_noun(name)
    normalized = normalize_entity_name(name, to_singular=True)

    if proper_noun:
        return normalized  # Preserve capitalization
    return normalized.title()  # Title case for schema names


def extract_resource_from_path(path: str) -> str:
    """Extract the primary resource name from an API path.

    Extracts the last meaningful path segment, ignoring path parameters.

    Args:
        path: API path (e.g., "/breeds", "/users/{id}/orders", "/api/v1/products")

    Returns:
        Resource name (e.g., "breeds", "orders", "products")
    """
    # Split path into segments
    segments = [s for s in path.split('/') if s]

    # Filter out path parameters ({id}, {userId}, etc.) and common prefixes
    ignore_patterns = {'api', 'v1', 'v2', 'v3'}
    meaningful_segments = []

    for segment in segments:
        # Skip path parameters
        if segment.startswith('{') and segment.endswith('}'):
            continue
        # Skip common API prefixes
        if segment.lower() in ignore_patterns:
            continue
        meaningful_segments.append(segment)

    if not meaningful_segments:
        # Fallback to last segment even if it's a parameter
        return segments[-1] if segments else path

    # Return the last meaningful segment (the primary resource)
    return meaningful_segments[-1]


class SemanticType:
    """Semantic type constants - linguistic role of the entity."""
    CONCEPT = "concept"      # Noun/thing: customer, order, user, product
    ATTRIBUTE = "attribute"  # Modifier/property: active, pending, total, name
    ACTION = "action"        # Verb/operation: create, get, delete, update
    TERM = "term"            # Compound phrase: "customer lifetime value", "MRR"


class NerType:
    """NER type constants - entity source classifier.

    Indicates where the entity was identified from:
    - spaCy NER: ORG, PERSON, PRODUCT, GPE, EVENT, LAW
    - Custom patterns: SCHEMA, API, TERM
    """
    # spaCy built-in NER
    ORG = "ORG"          # Organizations
    PERSON = "PERSON"    # People
    PRODUCT = "PRODUCT"  # Products
    GPE = "GPE"          # Geo-political entities
    EVENT = "EVENT"      # Events
    LAW = "LAW"          # Legal documents/references
    # Custom EntityRuler patterns
    SCHEMA = "SCHEMA"    # Database tables/columns
    API = "API"          # API endpoints/schemas
    TERM = "TERM"        # Business glossary terms


# Keep EntityType for backwards compatibility during migration
class EntityType:
    """Entity type constants. DEPRECATED: Use SemanticType and NerType instead."""
    # Database schema
    TABLE = "table"
    COLUMN = "column"
    # API schemas
    API_ENDPOINT = "api_endpoint"
    API_SCHEMA = "api_schema"
    GRAPHQL_TYPE = "graphql_type"
    GRAPHQL_FIELD = "graphql_field"
    # Business/domain
    CONCEPT = "concept"
    BUSINESS_TERM = "business_term"
    # spaCy NER types
    ORGANIZATION = "organization"
    PRODUCT = "product"
    LOCATION = "location"
    PERSON = "person"
    EVENT = "event"
    METRIC = "metric"


@dataclass
class EntityRelationship:
    """An SVO relationship triple between two entities (keyed by name)."""
    id: str
    subject_name: str
    verb: str
    object_name: str
    sentence: str = ""
    confidence: float = 1.0
    verb_category: str = "other"
    session_id: str = ""
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class GlossaryTerm:
    """A curated glossary term with business definition.

    Attributes:
        id: Primary key
        name: Singular canonical form (matches entity.name)
        display_name: Title case singular for UI
        definition: Business meaning
        domain: Owning domain (None = system-level)
        parent_id: Taxonomy parent (self-referential)
        aliases: Alternate names
        semantic_type: CONCEPT, ATTRIBUTE, ACTION, TERM
        cardinality: many | distinct | singular
        plural: Irregular plural form
        list_of: Glossary term ID if collection
        tags: Map of tags
        owner: Term-level owner
        status: draft | reviewed | approved
        provenance: llm | human | hybrid
        session_id: Session that owns this term
    """
    id: str
    name: str
    display_name: str
    definition: str
    domain: Optional[str] = None
    parent_id: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    semantic_type: Optional[str] = None
    cardinality: str = "many"
    plural: Optional[str] = None
    list_of: Optional[str] = None
    tags: dict[str, dict] = field(default_factory=dict)
    owner: Optional[str] = None
    status: str = "draft"
    provenance: str = "llm"
    session_id: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        now = datetime.now()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now


@dataclass
class Entity:
    """An NER-extracted entity from document chunks.

    Attributes:
        id: Primary key (hash of name, semantic_type, session_id)
        name: Normalized name for NER matching (lowercase, singular)
        display_name: Title case name for display ("Customer Order")
        semantic_type: Linguistic role (CONCEPT, ATTRIBUTE, ACTION, TERM)
        ner_type: spaCy NER type (ORG, PERSON, etc.) or None for schema/API patterns
        session_id: Session that owns this entity (required)
        domain_id: Domain this entity came from
    """
    id: str
    name: str  # Normalized: lowercase, singular
    display_name: str  # Title case for display
    semantic_type: str  # SemanticType value
    session_id: str
    ner_type: Optional[str] = None  # NerType value or None
    domain_id: Optional[str] = None
    created_at: Optional[datetime] = None

    # Backwards compatibility: expose 'type' as alias for semantic_type
    @property
    def type(self) -> str:
        return self.semantic_type

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ChunkEntity:
    """Link between a chunk and an NER-extracted entity."""
    chunk_id: str
    entity_id: str
    confidence: float = 1.0  # NER confidence
    mention_count: int = 1
    mention_text: str = ""


@dataclass
class EnrichedChunk:
    """A chunk with its similarity score and associated entities."""
    chunk: "DocumentChunk"
    score: float
    entities: list[Entity] = field(default_factory=list)


@dataclass
class DocumentChunk:
    """A chunk of a document for embedding and search.

    Attributes:
        document_name: Name of the document this chunk belongs to
        content: The text content of this chunk
        section: Optional section header this chunk is under
        chunk_index: Index of this chunk within the document
        source: Resource type - 'schema', 'api', or 'document'
        chunk_type: Granular type from ChunkType enum
    """
    document_name: str
    content: str
    section: Optional[str] = None
    chunk_index: int = 0
    source: str = "document"
    chunk_type: ChunkType = ChunkType.DOCUMENT


@dataclass
class LoadedDocument:
    """A loaded document with content and metadata."""
    name: str
    config: DocumentConfig
    content: str
    format: str
    sections: list[str] = field(default_factory=list)
    loaded_at: Optional[str] = None
    file_mtime: Optional[float] = None  # File modification time for change detection
    content_hash: Optional[str] = None  # Hash of content for change detection


@dataclass
class StructuredFileSchema:
    """Inferred schema for a structured data file."""
    filename: str
    filepath: str
    file_format: str  # csv, json, jsonl, parquet
    row_count: Optional[int] = None
    columns: list[dict] = field(default_factory=list)  # [{name, type, sample_values}]
    description: Optional[str] = None

    def to_metadata_doc(self) -> str:
        """Generate a metadata document for indexing."""
        lines = [
            f"Structured Data File: {self.filename}",
            f"Path: {self.filepath}",
            f"Format: {self.file_format}",
        ]
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.row_count is not None:
            lines.append(f"Row count: {self.row_count}")

        if self.columns:
            lines.append("\nColumns:")
            for col in self.columns:
                col_line = f"  - {col['name']} ({col.get('type', 'unknown')})"
                if col.get('sample_values'):
                    samples = col['sample_values'][:5]  # Limit samples
                    col_line += f": {samples}"
                lines.append(col_line)

        return "\n".join(lines)
