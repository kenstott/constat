# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SQLAlchemy ORM models for all relational tables.

Schema documentation for Phase 1 (DuckDB raw SQL).
Becomes the actual ORM in Phase 3 (PostgreSQL migration).

JSON fields (aliases, tags) use Text with manual serialization
for portability across backends.
"""

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    Text,
    TIMESTAMP,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class EntityModel(Base):
    __tablename__ = "entities"

    id = Column(Text, primary_key=True)
    name = Column(Text, nullable=False)
    display_name = Column(Text, nullable=False)
    semantic_type = Column(Text, nullable=False)
    ner_type = Column(Text)
    session_id = Column(Text, nullable=False)
    domain_id = Column(Text)
    created_at = Column(TIMESTAMP)


class ChunkEntityModel(Base):
    __tablename__ = "chunk_entities"

    chunk_id = Column(Text, primary_key=True)
    entity_id = Column(Text, primary_key=True)
    confidence = Column(Float, default=1.0)


class GlossaryTermModel(Base):
    __tablename__ = "glossary_terms"

    id = Column(Text, primary_key=True)
    name = Column(Text, nullable=False)
    display_name = Column(Text, nullable=False)
    definition = Column(Text, nullable=False)
    domain = Column(Text)
    parent_id = Column(Text)
    parent_verb = Column(Text, default="HAS_KIND")
    aliases = Column(Text)  # JSON serialized list
    semantic_type = Column(Text)
    cardinality = Column(Text, default="many")
    plural = Column(Text)
    tags = Column(Text)  # JSON serialized dict
    owner = Column(Text)
    status = Column(Text, default="draft")
    provenance = Column(Text, default="llm")
    session_id = Column(Text, nullable=False)
    user_id = Column(Text, nullable=False, default="default")
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
    ignored = Column(Boolean, default=False)


class EntityRelationshipModel(Base):
    __tablename__ = "entity_relationships"

    id = Column(Text, primary_key=True)
    subject_name = Column(Text, nullable=False)
    verb = Column(Text, nullable=False)
    object_name = Column(Text, nullable=False)
    sentence = Column(Text)
    confidence = Column(Float, default=1.0)
    verb_category = Column(Text, default="other")
    session_id = Column(Text)
    user_edited = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP)

    __table_args__ = (
        UniqueConstraint("subject_name", "verb", "object_name", "session_id"),
    )


class GlossaryClusterModel(Base):
    __tablename__ = "glossary_clusters"

    term_name = Column(Text, primary_key=True)
    cluster_id = Column(Integer, nullable=False)
    session_id = Column(Text, primary_key=True)


class SourceHashModel(Base):
    __tablename__ = "source_hashes"

    source_id = Column(Text, primary_key=True)
    db_hash = Column(Text)
    api_hash = Column(Text)
    doc_hash = Column(Text)
    updated_at = Column(TIMESTAMP)


class ResourceHashModel(Base):
    __tablename__ = "resource_hashes"

    resource_id = Column(Text, primary_key=True)
    resource_type = Column(Text, nullable=False)
    resource_name = Column(Text, nullable=False)
    source_id = Column(Text, nullable=False)
    content_hash = Column(Text, nullable=False)
    updated_at = Column(TIMESTAMP)


class DocumentUrlModel(Base):
    __tablename__ = "document_urls"

    document_name = Column(Text, primary_key=True)
    source_url = Column(Text, nullable=False)

