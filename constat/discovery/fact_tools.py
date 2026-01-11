"""Fact resolution tools for on-demand knowledge lookup.

These tools integrate with the FactResolver and document discovery
to provide multi-source fact resolution with confidence scoring.
"""

from typing import Optional, Any

from constat.execution.fact_resolver import FactResolver, Fact, FactSource
from constat.discovery.doc_tools import DocumentDiscoveryTools


class FactResolutionTools:
    """Tools for resolving facts from multiple sources on-demand."""

    def __init__(
        self,
        fact_resolver: Optional[FactResolver] = None,
        doc_tools: Optional[DocumentDiscoveryTools] = None,
    ):
        self.fact_resolver = fact_resolver
        self.doc_tools = doc_tools

    def resolve_fact(self, question: str) -> dict:
        """
        Resolve a factual question using all available sources.

        Sources checked (in priority order):
        1. Cache (previously resolved facts)
        2. Config (system prompt, settings)
        3. Documents (reference docs, business rules)
        4. Rules (registered derivation rules)
        5. Database (LLM-generated queries)
        6. LLM knowledge (world facts, heuristics)

        Args:
            question: Natural language question (e.g., "What defines a VIP customer?")

        Returns:
            Dict with answer, confidence, and sources
        """
        sources_checked = []
        answer = None
        confidence = 0.0
        source_info = []

        # Try document search first (often has business rules)
        if self.doc_tools:
            doc_results = self.doc_tools.search_documents(question, limit=3)
            sources_checked.append("documents")

            if doc_results and doc_results[0].get("relevance", 0) > 0.7:
                # Good document match - use as answer
                top_result = doc_results[0]
                answer = top_result.get("excerpt", "")
                confidence = top_result.get("relevance", 0.8)
                source_info.append({
                    "type": "document",
                    "name": top_result.get("document"),
                    "excerpt": answer,
                    "confidence": confidence,
                })

        # Try fact resolver if available and no good document answer
        if self.fact_resolver and (answer is None or confidence < 0.8):
            # Convert question to fact name (simplified)
            fact_name = self._question_to_fact_name(question)
            sources_checked.append("fact_resolver")

            fact = self.fact_resolver.resolve(fact_name)
            if fact.is_resolved:
                # Use fact resolver result if better
                if fact.confidence > confidence:
                    answer = str(fact.value)
                    confidence = fact.confidence
                    source_info.append({
                        "type": fact.source.value,
                        "name": fact.name,
                        "value": fact.value,
                        "confidence": fact.confidence,
                        "reasoning": fact.reasoning,
                    })

        # Determine if clarification needed
        needs_clarification = confidence < 0.6 or answer is None

        return {
            "question": question,
            "answer": answer or "Unable to resolve this fact from available sources.",
            "confidence": round(confidence, 2),
            "sources": source_info,
            "sources_checked": sources_checked,
            "needs_clarification": needs_clarification,
        }

    def add_fact(self, name: str, value: Any, reasoning: Optional[str] = None) -> dict:
        """
        Add a user-provided fact to the cache.

        Args:
            name: Fact identifier (e.g., "vip_threshold")
            value: The fact value
            reasoning: Optional explanation

        Returns:
            Dict with the added fact details
        """
        if not self.fact_resolver:
            return {"error": "Fact resolver not configured"}

        fact = self.fact_resolver.add_user_fact(name, value, reasoning)

        return {
            "name": fact.name,
            "value": fact.value,
            "confidence": fact.confidence,
            "source": fact.source.value,
            "reasoning": fact.reasoning,
        }

    def extract_facts_from_text(self, text: str) -> dict:
        """
        Extract and add facts from natural language text.

        Parses statements like:
        - "The VIP threshold is $100,000"
        - "There were 1 million attendees"

        Args:
            text: Natural language text containing facts

        Returns:
            Dict with extracted facts
        """
        if not self.fact_resolver:
            return {"error": "Fact resolver not configured", "facts": []}

        facts = self.fact_resolver.add_user_facts_from_text(text)

        return {
            "extracted_count": len(facts),
            "facts": [
                {
                    "name": f.name,
                    "value": f.value,
                    "reasoning": f.reasoning,
                }
                for f in facts
            ],
        }

    def list_known_facts(self) -> dict:
        """
        List all facts currently known (cached/resolved).

        Returns:
            Dict with list of known facts and their sources
        """
        facts = []

        if self.fact_resolver:
            for key, fact in self.fact_resolver._cache.items():
                facts.append({
                    "name": key,
                    "value": fact.value,
                    "confidence": fact.confidence,
                    "source": fact.source.value,
                })

        return {
            "count": len(facts),
            "facts": facts,
        }

    def get_unresolved_facts(self) -> dict:
        """
        Get facts that could not be resolved.

        Returns:
            Dict with unresolved facts and suggestions
        """
        if not self.fact_resolver:
            return {"unresolved": [], "suggestions": ""}

        unresolved = self.fact_resolver.get_unresolved_facts()

        return {
            "unresolved": [
                {
                    "name": f.name,
                    "reasoning": f.reasoning,
                }
                for f in unresolved
            ],
            "suggestions": self.fact_resolver.get_unresolved_summary(),
        }

    def _question_to_fact_name(self, question: str) -> str:
        """Convert a natural language question to a fact name."""
        # Simple heuristic - extract key terms
        question = question.lower()

        # Remove common question words
        for word in ["what", "is", "are", "the", "a", "an", "how", "many", "much", "does", "do", "?"]:
            question = question.replace(word, " ")

        # Clean up and join
        words = question.split()
        if len(words) > 5:
            words = words[:5]

        return "_".join(words).strip("_")


# Tool schemas for LLM
FACT_TOOL_SCHEMAS = [
    {
        "name": "resolve_fact",
        "description": "Resolve a factual question using all available sources including documents, database, and LLM knowledge. Returns answer with confidence score and provenance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question to resolve (e.g., 'What defines a VIP customer?', 'What is the revenue threshold?')",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "add_fact",
        "description": "Add a user-provided fact to the knowledge base. Use when the user provides information that should be remembered.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Fact identifier (e.g., 'vip_threshold', 'target_revenue')",
                },
                "value": {
                    "type": ["string", "number", "boolean"],
                    "description": "The fact value",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Optional explanation of where this fact came from",
                },
            },
            "required": ["name", "value"],
        },
    },
    {
        "name": "extract_facts_from_text",
        "description": "Extract and store facts from natural language text. Use when the user provides multiple facts in conversational form.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Natural language text containing facts to extract",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "list_known_facts",
        "description": "List all facts currently known from previous queries and user input. Use to check what information is already available.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_unresolved_facts",
        "description": "Get facts that could not be resolved and suggestions for how to provide them. Use when a query failed due to missing information.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
