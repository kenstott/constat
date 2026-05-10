# Copyright (c) 2025 Kenneth Stott
# Canary: 6cdfbafd-6293-4afe-a592-4e8ff6ddc811
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Build vector store chunks from relationship config sections and glossary terms.

Converts relationship definitions into DocumentChunk objects for embedding
and semantic search. Glossary terms are built from DB-sourced GlossaryTerm objects.
"""

import logging
from typing import Any, TYPE_CHECKING

from constat.discovery.models import ChunkType, DocumentChunk

if TYPE_CHECKING:
    from constat.discovery.models import GlossaryTerm

logger = logging.getLogger(__name__)


def build_relationship_chunks(relationships: dict[str, Any]) -> list[DocumentChunk]:
    """Build chunks from relationships config section.

    Each relationship becomes a chunk with the relationship name,
    definition, aliases, and connected entities.

    Expected relationships format:
        relationships:
          customer_orders:
            definition: A customer places orders
            aliases: [purchase history, order history]
            entities: [customer, order]
          order_products:
            definition: An order contains products
            entities: [order, product, line_item]

    Args:
        relationships: Relationships config dict

    Returns:
        List of DocumentChunks with chunk_type=RELATIONSHIP
    """
    chunks = []
    for idx, (name, value) in enumerate(relationships.items()):
        if isinstance(value, str):
            definition = value
            aliases = []
            entities = []
        elif isinstance(value, dict):
            definition = value.get("definition", "")
            aliases = value.get("aliases", [])
            entities = value.get("entities", [])
        else:
            continue

        # Build rich text content for embedding
        lines = [f"Relationship: {name}"]
        if definition:
            lines.append(f"Definition: {definition}")
        if entities:
            lines.append(f"Connects: {', '.join(entities)}")
        if aliases:
            lines.append(f"Also known as: {', '.join(aliases)}")

        content = "\n".join(lines)

        chunks.append(DocumentChunk(
            document_name=f"relationship:{name}",
            content=content,
            section="relationships",
            chunk_index=idx,
            source="relationships",
            chunk_type=ChunkType.RELATIONSHIP,
        ))

    logger.debug(f"Built {len(chunks)} relationship chunks")
    return chunks


def glossary_term_to_chunk(
    term: "GlossaryTerm",
    entity_sources: list[str],
    relationships: list[dict] | None = None,
) -> DocumentChunk:
    """Build an embeddable chunk from a GlossaryTerm with its physical resources.

    Args:
        term: GlossaryTerm dataclass instance
        entity_sources: Resolved physical resource strings (e.g., "crm.customers (schema)")
        relationships: SVO triples from entity_relationships (dicts with subject_name, verb, object_name)

    Returns:
        DocumentChunk with chunk_type=GLOSSARY_TERM
    """
    parts = [f"{term.display_name}: {term.definition}"]

    if term.aliases:
        parts.append(f"Also known as: {', '.join(term.aliases)}")

    if term.parent_id:
        parts.append(f"Parent: {term.parent_id}")

    if entity_sources:
        parts.append("Connected resources:")
        for source in entity_sources:
            parts.append(f"  - {source}")

    if relationships:
        parent_name = term.parent_id.lower() if term.parent_id else None

        rel_lines = []
        for rel in relationships:
            verb_upper = rel["verb"].upper().replace(" ", "_")
            if verb_upper.startswith("HAS_") and parent_name:
                other = rel["object_name"] if rel["subject_name"].lower() == term.name.lower() else rel["subject_name"]
                if other.lower() == parent_name:
                    continue
            rel_lines.append(f"{rel['subject_name']} {verb_upper} {rel['object_name']}")

        if rel_lines:
            parts.append("Related: " + "; ".join(rel_lines))

    return DocumentChunk(
        document_name=f"glossary:{term.id}",
        content="\n".join(parts),
        section="glossary",
        chunk_index=0,
        source="document",
        chunk_type=ChunkType.GLOSSARY_TERM,
    )


def get_relationship_terms_for_ner(relationships: dict[str, Any]) -> list[str]:
    """Extract relationship names and aliases for NER entity ruler.

    Args:
        relationships: Relationships config dict

    Returns:
        List of terms to add as business_terms in EntityExtractor
    """
    terms = []
    for name, value in relationships.items():
        terms.append(name.replace("_", " "))
        if isinstance(value, dict):
            for alias in value.get("aliases", []):
                terms.append(alias)
    return terms
