# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Fine-tuning exemplar generator.

Exports conversation pairs for fine-tuning at three coverage levels:
- minimal: high-confidence, high-applied rules only (frontier models)
- standard: all rules + approved/human glossary terms (mid-tier models)
- comprehensive: everything including all glossary terms + relationships (small models)

Outputs both OpenAI messages JSONL and Alpaca JSONL formats.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from constat.storage.learnings import LearningStore

logger = logging.getLogger(__name__)

COVERAGE_LEVELS = ("minimal", "standard", "comprehensive")


@dataclass
class ExemplarResult:
    """Result of exemplar generation."""
    rule_pairs: int = 0
    glossary_pairs: int = 0
    relationship_pairs: int = 0
    total: int = 0
    output_paths: dict[str, str] = field(default_factory=dict)


class ExemplarGenerator:
    """Generates fine-tuning exemplar pairs from rules, glossary, and relationships."""

    RULE_BATCH_SIZE = 10
    GLOSSARY_BATCH_SIZE = 10
    RELATIONSHIP_BATCH_SIZE = 15

    def __init__(
        self,
        learning_store: LearningStore,
        vector_store,
        llm,
        session_id: str,
        user_id: str,
    ):
        self.store = learning_store
        self.vector_store = vector_store
        self.llm = llm
        self.session_id = session_id
        self.user_id = user_id
        self.output_dir = Path(".constat") / user_id

    def generate(self, coverage: str = "standard") -> ExemplarResult:
        """Generate exemplars at the given coverage level."""
        if coverage not in COVERAGE_LEVELS:
            raise ValueError(f"coverage must be one of {COVERAGE_LEVELS}")

        rules = self._select_rules(coverage)
        terms = self._select_glossary_terms(coverage)
        relationships = self._select_relationships(coverage)

        logger.info(
            f"[exemplar] coverage={coverage} rules={len(rules)} "
            f"terms={len(terms)} rels={len(relationships)}"
        )

        exemplars: list[dict] = []

        if rules:
            rule_exemplars = self._generate_rule_exemplars(rules)
            exemplars.extend(rule_exemplars)
            rule_count = len(rule_exemplars)
        else:
            rule_count = 0

        if terms:
            glossary_exemplars = self._generate_glossary_exemplars(terms)
            exemplars.extend(glossary_exemplars)
            glossary_count = len(glossary_exemplars)
        else:
            glossary_count = 0

        if relationships:
            rel_exemplars = self._generate_relationship_exemplars(relationships)
            exemplars.extend(rel_exemplars)
            rel_count = len(rel_exemplars)
        else:
            rel_count = 0

        output_paths = self._write_jsonl(exemplars)

        result = ExemplarResult(
            rule_pairs=rule_count,
            glossary_pairs=glossary_count,
            relationship_pairs=rel_count,
            total=len(exemplars),
            output_paths=output_paths,
        )

        self.store.save_exemplar_run({
            "coverage": coverage,
            "rule_pairs": rule_count,
            "glossary_pairs": glossary_count,
            "relationship_pairs": rel_count,
            "total": len(exemplars),
            "paths": output_paths,
        })

        return result

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_rules(self, coverage: str) -> list[dict]:
        if coverage == "minimal":
            rules = self.store.list_rules(min_confidence=0.8)
            return [r for r in rules if r.get("applied_count", 0) >= 3]
        # standard and comprehensive: all rules
        return self.store.list_rules()

    def _select_glossary_terms(self, coverage: str):
        if coverage == "minimal":
            return []
        terms = self.vector_store.list_glossary_terms(
            self.session_id, user_id=self.user_id,
        )
        if coverage == "standard":
            return [
                t for t in terms
                if t.status == "approved" or t.provenance == "human"
            ]
        # comprehensive: all defined terms
        return [t for t in terms if t.definition]

    def _select_relationships(self, coverage: str):
        if coverage != "comprehensive":
            return []
        rows = self.vector_store._conn.execute(
            "SELECT id, subject_name, verb, object_name, sentence, confidence "
            "FROM entity_relationships WHERE session_id = ? "
            "ORDER BY confidence DESC",
            [self.session_id],
        ).fetchall()
        return [
            {
                "id": r[0], "subject_name": r[1], "verb": r[2],
                "object_name": r[3], "sentence": r[4] or "", "confidence": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # LLM generation (batched)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(response: str) -> list[dict]:
        content = response.strip() if isinstance(response, str) else response.content.strip()
        if "```" in content:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)
        return json.loads(content)

    def _generate_rule_exemplars(self, rules: list[dict]) -> list[dict]:
        exemplars = []
        for i in range(0, len(rules), self.RULE_BATCH_SIZE):
            batch = rules[i:i + self.RULE_BATCH_SIZE]
            numbered = "\n".join(
                f'{j}. "{r["summary"]}" [tags: {", ".join(r.get("tags", []))}]'
                for j, r in enumerate(batch)
            )
            prompt = (
                "Generate training conversation pairs for these coding rules.\n"
                "Each pair: a realistic user question + correct assistant response demonstrating the rule.\n"
                "2-3 pairs per rule.\n\n"
                f"Rules:\n{numbered}\n\n"
                'JSON array: [{"rule_index": 0, "user": "...", "assistant": "..."}]'
            )
            try:
                response = self.llm.generate(
                    system="You generate fine-tuning exemplar pairs from coding rules.",
                    user_message=prompt,
                    max_tokens=self.llm.max_output_tokens,
                )
                pairs = self._extract_json(response)
                for p in pairs:
                    exemplars.append({"user": p["user"], "assistant": p["assistant"]})
            except Exception as e:
                logger.warning(f"[exemplar] rule batch failed: {e}")
        return exemplars

    def _generate_glossary_exemplars(self, terms) -> list[dict]:
        exemplars = []
        for i in range(0, len(terms), self.GLOSSARY_BATCH_SIZE):
            batch = terms[i:i + self.GLOSSARY_BATCH_SIZE]
            numbered = "\n".join(
                f'{j}. {t.name} — {t.definition} (aliases: {", ".join(t.aliases)})'
                for j, t in enumerate(batch)
            )
            prompt = (
                "Generate training pairs for these domain terms.\n"
                "For each: (1) user asks about concept -> assistant uses canonical term,\n"
                "(2) user uses an alias -> assistant naturally corrects to canonical name.\n\n"
                f"Terms:\n{numbered}\n\n"
                'JSON array: [{"term_index": 0, "user": "...", "assistant": "..."}]'
            )
            try:
                response = self.llm.generate(
                    system="You generate fine-tuning exemplar pairs from domain glossary terms.",
                    user_message=prompt,
                    max_tokens=self.llm.max_output_tokens,
                )
                pairs = self._extract_json(response)
                for p in pairs:
                    exemplars.append({"user": p["user"], "assistant": p["assistant"]})
            except Exception as e:
                logger.warning(f"[exemplar] glossary batch failed: {e}")
        return exemplars

    def _generate_relationship_exemplars(self, rels: list[dict]) -> list[dict]:
        exemplars = []
        for i in range(0, len(rels), self.RELATIONSHIP_BATCH_SIZE):
            batch = rels[i:i + self.RELATIONSHIP_BATCH_SIZE]
            numbered = "\n".join(
                f'{j}. {r["subject_name"]} -> {r["verb"]} -> {r["object_name"]}: {r["sentence"]}'
                for j, r in enumerate(batch)
            )
            prompt = (
                "Generate a training pair for each relationship showing correct reasoning.\n\n"
                f"Relationships:\n{numbered}\n\n"
                'JSON array: [{"rel_index": 0, "user": "...", "assistant": "..."}]'
            )
            try:
                response = self.llm.generate(
                    system="You generate fine-tuning exemplar pairs from entity relationships.",
                    user_message=prompt,
                    max_tokens=self.llm.max_output_tokens,
                )
                pairs = self._extract_json(response)
                for p in pairs:
                    exemplars.append({"user": p["user"], "assistant": p["assistant"]})
            except Exception as e:
                logger.warning(f"[exemplar] relationship batch failed: {e}")
        return exemplars

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _write_jsonl(self, exemplars: list[dict]) -> dict[str, str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        messages_path = self.output_dir / "exemplars_messages.jsonl"
        alpaca_path = self.output_dir / "exemplars_alpaca.jsonl"

        system_content = "You are a helpful data analyst assistant."

        with open(messages_path, "w") as f:
            for ex in exemplars:
                line = {
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": ex["user"]},
                        {"role": "assistant", "content": ex["assistant"]},
                    ]
                }
                f.write(json.dumps(line) + "\n")

        with open(alpaca_path, "w") as f:
            for ex in exemplars:
                line = {
                    "instruction": ex["user"],
                    "input": "",
                    "output": ex["assistant"],
                }
                f.write(json.dumps(line) + "\n")

        return {
            "messages": str(messages_path),
            "alpaca": str(alpaca_path),
        }
