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

from constat.discovery.models import (
    DocumentChunk,
    Entity,
    ChunkEntity,
    SemanticType,
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

    # Markdown/wiki noise patterns
    URL_ENCODED_RE = re.compile(r'%[0-9A-Fa-f]{2}')
    WIKI_NOISE_RE = re.compile(r'=Edit|#Cite|Citeref|Redlink|/[Ww]iki/', re.I)
    DATE_RE = re.compile(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$|^\d{1,2}\s+\w+\s+\d{4}$')
    BARE_NUMBER_RE = re.compile(r'^\d+$')
    PAREN_URL_RE = re.compile(r'^\(/')
    CSS_HTML_RE = re.compile(r'[;{}]|!Important|@Media', re.I)
    CHAPTER_SECTION_RE = re.compile(r'^(Chapter|Section|Article|Title|Page)\s+\d', re.I)
    LEADING_ARTICLE_RE = re.compile(r'^(the|a|an)\s+', re.I)
    STARTS_WITH_DIGIT_RE = re.compile(r'^\d+\s')
    TRAILING_FRAGMENT_RE = re.compile(r'\s+(of|de|du|des|di|s|the|and|&|for|in|on|to|vs|v)\s*$', re.I)
    UNMATCHED_PAREN_RE = re.compile(r'\([^)]*$|^[^(]*\)')
    WIKI_SITE_TERMS = frozenset({
        'wikipedia', 'wikidata', 'wikimedia', 'wikisource',
        'wikiversity', 'wikivoyage', 'wiktionary', 'wikibook', 'wiki project',
    })

    # Common noise words
    NOISE_WORDS = {
        'level', 'type', 'status', 'category', 'name', 'email', 'phone',
        'address', 'date', 'time', 'id', 'page', 'section', 'table',
    }

    def __init__(
        self,
        session_id: str,
        domain_id: Optional[str] = None,
        schema_terms: Optional[list[str]] = None,
        api_terms: Optional[list[str]] = None,
        business_terms: Optional[list[str]] = None,
        stop_list: Optional[set[str]] = None,
    ):
        """Initialize the entity extractor.

        Args:
            session_id: Session ID (required - entities are session-scoped)
            domain_id: Optional domain ID
            schema_terms: Database table/column names to recognize
            api_terms: API endpoint names to recognize
            business_terms: Business glossary terms to recognize
            stop_list: Additional terms to filter out during extraction
        """
        self.session_id = session_id
        self.domain_id = domain_id
        self._stop_list = stop_list or set()

        # Build custom NLP pipeline with EntityRuler
        self._nlp = get_nlp()

        # Add EntityRuler for custom patterns (before NER to take precedence)
        if "custom_ruler" in self._nlp.pipe_names:
            self._nlp.remove_pipe("custom_ruler")

        ruler = self._nlp.add_pipe("entity_ruler", name="custom_ruler", before="ner")

        patterns = []

        def make_token_pattern(text: str, use_lemma: bool = True) -> list[dict]:
            """Create a token pattern for matching.

            Args:
                text: Space-separated text to match
                use_lemma: If True, use LEMMA (singular/plural). If False, use LOWER only.
            """
            words = re.split(r'[\s_]+', text)
            if use_lemma:
                # LEMMA handles singular/plural but preserves case, so we also need LOWER
                # Using both: LEMMA for normalization, but match on lowercased lemma
                return [{"LEMMA": {"IN": [w.lower(), w.lower().rstrip('s')]}} for w in words if w]
            else:
                return [{"LOWER": w.lower()} for w in words if w]

        # Add schema patterns
        for term in (schema_terms or []):
            if term and len(term) > 1:
                # Normalize the term (converts underscores/camelCase to spaces)
                normalized = normalize_entity_name(term, to_singular=False)
                singular = normalize_entity_name(term, to_singular=True)

                # 1. Multi-token pattern for singular (matches "performance review")
                singular_pattern = [{"LOWER": w.lower()} for w in singular.split() if w]
                if singular_pattern:
                    patterns.append({"label": "SCHEMA", "pattern": singular_pattern})

                # 2. Multi-token pattern for plural if different (matches "performance reviews")
                if normalized != singular:
                    plural_pattern = [{"LOWER": w.lower()} for w in normalized.split() if w]
                    if plural_pattern:
                        patterns.append({"label": "SCHEMA", "pattern": plural_pattern})

                # 3. Single-token pattern for underscore terms (matches "performance_reviews")
                if '_' in term:
                    patterns.append({"label": "SCHEMA", "pattern": [{"LOWER": term.lower()}]})

                # 4. Single-token patterns for camelCase/PascalCase joined forms
                if ' ' in singular:
                    # Singular joined (matches "performance review")
                    singular_joined = singular.replace(' ', '').lower()
                    patterns.append({"label": "SCHEMA", "pattern": [{"LOWER": singular_joined}]})
                    # Plural joined if different (matches "performance reviews")
                    if normalized != singular:
                        plural_joined = normalized.replace(' ', '').lower()
                        if plural_joined != singular_joined:
                            patterns.append({"label": "SCHEMA", "pattern": [{"LOWER": plural_joined}]})

        # Add API patterns (case-insensitive token matching)
        for term in (api_terms or []):
            if term and len(term) > 1:
                token_pattern = make_token_pattern(term)
                if token_pattern:
                    patterns.append({"label": "API", "pattern": token_pattern})

        # Add business term patterns (case-insensitive token matching)
        for term in (business_terms or []):
            if term and len(term) > 1:
                token_pattern = make_token_pattern(term)
                if token_pattern:
                    patterns.append({"label": "TERM", "pattern": token_pattern})

        if patterns:
            # noinspection PyUnresolvedReferences
            ruler.add_patterns(patterns)
            logger.debug(f"EntityExtractor: added {len(patterns)} custom patterns")

        # Entity cache for deduplication
        self._entity_cache: dict[str, Entity] = {}

    def _generate_entity_id(self, name: str) -> str:
        """Generate entity ID from normalized name + session."""
        normalized = normalize_entity_name(name).lower()
        return hashlib.sha256(f"{normalized}:{self.session_id}".encode()).hexdigest()[:16]

    # Hex hash pattern (8+ hex chars, no spaces)
    HEX_PATTERN = re.compile(r'^[0-9a-fA-F]{8,}$')

    @staticmethod
    def _clean_for_ner(text: str) -> str:
        """Strip markdown/wiki syntax from text before NER processing."""
        # Remove wiki templates: {{...}}
        text = re.sub(r'\{\{[^}]*\}\}', '', text)
        # Remove HTML tags: <ref>...</ref>, <sup>, etc.
        text = re.sub(r'<[^>]+>', '', text)
        # Remove wiki-style links: [[display|target]] → display, [[text]] → text
        text = re.sub(r'\[\[([^|\]]*)\|([^\]]*)\]\]', r'\1', text)
        text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
        # Remove images: ![alt](url)
        text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
        # Remove links: [text](url) → text
        text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
        # Remove bare bracketed URLs: [url] (no display text pattern)
        text = re.sub(r'\[https?://[^\]]*\]', '', text)
        # Remove bold/italic: **text** or *text* or __text__ or _text_
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
        text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
        # Remove headings: # Heading → Heading
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Remove backtick code spans: `code` → code
        text = re.sub(r'`([^`]*)`', r'\1', text)
        # Remove ≈ comparison markers from wiki tables
        text = re.sub(r'≈\s*', '', text)
        return text

    def _is_noise(self, text: str) -> bool:
        """Check if text is noise (symbols, too short, etc.)."""
        # Require at least 2 ASCII alpha characters (filters non-English fragments)
        ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
        if ascii_alpha < 2:
            return True
        if self.NOISE_PATTERN.search(text):
            return True
        if text.lower() in self.NOISE_WORDS:
            return True
        if text.lower() in self._stop_list:
            return True
        # Skip if mostly non-alphanumeric
        alnum_count = sum(1 for c in text if c.isalnum())
        if alnum_count < len(text) * 0.5:
            return True
        # URL-encoded strings
        if self.URL_ENCODED_RE.search(text):
            return True
        # Wiki noise fragments
        if self.WIKI_NOISE_RE.search(text):
            return True
        # Bare dates
        if self.DATE_RE.match(text):
            return True
        # Bare numbers
        if self.BARE_NUMBER_RE.match(text):
            return True
        # Citation anchors
        if text.startswith('^'):
            return True
        # Parenthesized URL paths
        if self.PAREN_URL_RE.match(text):
            return True
        # CSS/HTML fragments (font declarations, cursor styles, media queries)
        if self.CSS_HTML_RE.search(text):
            return True
        # Wikipedia site references
        text_lower = text.lower()
        if any(w in text_lower for w in self.WIKI_SITE_TERMS):
            return True
        # Wikipedia template parameter separators
        if '","' in text:
            return True
        # Starts with ≈ (wiki table comparison markers)
        if text.startswith('≈'):
            return True
        # Leading/trailing quotes (broken markdown fragments)
        if text.startswith('"') or text.startswith('\\"') or text.endswith('"'):
            return True
        # Wiki action links
        if '&action' in text_lower:
            return True
        # Chapter/Section/Article + number references
        if self.CHAPTER_SECTION_RE.match(text):
            return True
        # Trailing asterisk (footnote markers)
        if text.endswith('*'):
            return True
        # Trailing single quote (possessive fragments)
        if text.endswith("'"):
            return True
        # Wrapped in single quotes
        if text.startswith("'") and text.endswith("'"):
            return True
        # Starts with & (markup fragments)
        if text.startswith('&'):
            return True
        # Starts with digit + space (e.g., "5 Type of Company Merger")
        if self.STARTS_WITH_DIGIT_RE.match(text):
            return True
        # Contains pipe (wiki table remnants)
        if '|' in text:
            return True
        # Contains slash (wiki paths, hybrid terms) — checked separately for non-API entities
        # Contains backslash
        if '\\' in text:
            return True
        # Contains › (wiki breadcrumb separator)
        if '›' in text:
            return True
        # Trailing preposition/conjunction (truncated phrases)
        if self.TRAILING_FRAGMENT_RE.search(text):
            return True
        # Unmatched parentheses (truncated references)
        if self.UNMATCHED_PAREN_RE.search(text):
            return True
        # Dot-notation schema paths (e.g., "Countries.Language.Rtl")
        if '.' in text:
            return True
        # Colon-separated labels (e.g., "Analytic Datum: Information")
        if ':' in text:
            return True
        # Hex hashes / auto-generated IDs
        clean = text.replace('-', '').replace('_', '')
        if self.HEX_PATTERN.match(clean):
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
        semantic_type = self._label_to_semantic_type(spacy_label, name)

        # Store the label as ner_type for all entities (spaCy + custom patterns)
        # This captures the entity source: ORG, PERSON, SCHEMA, API, TERM
        ner_type = spacy_label

        entity = Entity(
            id=self._generate_entity_id(name),
            name=normalized,  # Normalized for NER matching
            display_name=display_entity_name(name),  # Title case for display
            semantic_type=semantic_type,
            ner_type=ner_type,
            session_id=self.session_id,
            domain_id=self.domain_id,
        )
        self._entity_cache[normalized] = entity
        return entity

    # HTTP verbs that indicate actions (PUT, DELETE, PATCH are mutations)
    _ACTION_HTTP_VERBS = {'put', 'delete', 'patch'}

    @staticmethod
    def _label_to_semantic_type(label: str, name: str = "") -> str:
        """Map spaCy/custom label to semantic type.

        Args:
            label: Entity label (e.g., 'ORG', 'SCHEMA', 'API', 'TERM')
            name: Entity name (used for HTTP verb detection)

        Returns:
            SemanticType value
        """
        # Custom pattern labels
        if label == 'SCHEMA':
            return SemanticType.CONCEPT  # Tables, columns are things
        elif label == 'API':
            # Check for HTTP verbs — GET is a concept, POST is ambiguous (concept),
            # PUT/DELETE/PATCH are actions
            first_word = name.split()[0].lower() if name else ""
            if first_word in EntityExtractor._ACTION_HTTP_VERBS:
                return SemanticType.ACTION
            return SemanticType.CONCEPT
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
        clean = self._clean_for_ner(chunk.content)
        doc = self._nlp(clean)
        chunk_id = self._generate_chunk_id(chunk)

        results: list[tuple[Entity, ChunkEntity]] = []
        seen_entities: set[str] = set()

        for ent in doc.ents:
            text = ent.text.strip()

            # Strip leading articles (the/a/an) from entity text
            text = self.LEADING_ARTICLE_RE.sub('', text).strip()

            # Skip noise
            if self._is_noise(text):
                continue

            # Only keep relevant entity types
            if ent.label_ not in self.KEEP_SPACY_TYPES and ent.label_ not in {'SCHEMA', 'API', 'TERM'}:
                continue

            # Single-word PERSON entities are almost always noise (first name only)
            if ent.label_ == 'PERSON' and ' ' not in text:
                continue

            # Slash in non-API entities (wiki paths, hybrid terms)
            if '/' in text and ent.label_ not in {'API', 'SCHEMA', 'TERM'}:
                continue

            # Split dot-notation names into individual entities
            texts = text.split('.') if '.' in text else [text]

            for t in texts:
                t = t.strip()
                if not t or self._is_noise(t):
                    continue

                # Deduplicate within chunk
                normalized = normalize_entity_name(t).lower()
                if normalized in seen_entities:
                    continue
                seen_entities.add(normalized)

                entity = self._get_or_create_entity(t, ent.label_)

                link = ChunkEntity(
                    chunk_id=chunk_id,
                    entity_id=entity.id,
                    confidence=0.9 if ent.label_ in {'SCHEMA', 'API', 'TERM'} else 0.75,
                )

                results.append((entity, link))

        return results

    @staticmethod
    def _generate_chunk_id(chunk: DocumentChunk) -> str:
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
