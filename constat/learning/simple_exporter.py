# Copyright (c) 2025 Kenneth Stott
# Canary: b56b5858-fdc3-42af-b84c-ab6c591d8807
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Simple exemplar exporter — deterministic, no LLM calls.

Converts raw learnings, rules, and glossary terms into JSONL
in standard fine-tuning formats (OpenAI Messages, Alpaca, ShareGPT).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from constat.storage.learnings import LearningStore, LearningCategory


class SimpleExporter:
    """Export learnings/rules/glossary as fine-tuning JSONL."""

    def __init__(self, learning_store: LearningStore, vector_store=None):
        self.store = learning_store
        self.vector_store = vector_store

    def export(
        self,
        include: list[str] | None = None,
        fmt: Literal["messages", "alpaca", "sharegpt"] = "messages",
        domain: str | None = None,
        min_confidence: float = 0.0,
        since: datetime | None = None,
    ) -> str:
        """Export learnings as JSONL string.

        Args:
            include: Data types to include (corrections, rules, glossary)
            fmt: Output format
            domain: Filter by domain (None = all)
            min_confidence: Minimum confidence for rules
            since: Only include items created after this time

        Returns:
            JSONL string
        """
        include = include or ["corrections", "rules"]
        records: list[dict] = []

        if "corrections" in include:
            records.extend(self._corrections_to_records(domain, since))

        if "rules" in include:
            records.extend(self._rules_to_records(domain, min_confidence, since))

        if "glossary" in include:
            records.extend(self._glossary_to_records(domain))

        # Format
        if fmt == "alpaca":
            formatted = [self._to_alpaca(r) for r in records]
        elif fmt == "sharegpt":
            formatted = [self._to_sharegpt(r) for r in records]
        else:
            formatted = [self._to_messages(r) for r in records]

        return "\n".join(json.dumps(item, ensure_ascii=False) for item in formatted)

    def _corrections_to_records(self, domain: str | None, since: datetime | None) -> list[dict]:
        """Convert raw learnings to records."""
        learnings = self.store.list_raw_learnings(limit=500, include_promoted=True)
        records = []
        for l in learnings:
            if since:
                created = l.get("created", "")
                if created and created < since.isoformat():
                    continue

            ctx = l.get("context", {})
            user_msg = ctx.get("step_goal", "") or ctx.get("query_text", "")
            if not user_msg:
                user_msg = ctx.get("error_message", "")[:200]
            if not user_msg:
                continue

            correction = l.get("correction", "")
            if not correction:
                continue

            records.append({
                "system": "You are a data analyst assistant.",
                "user": user_msg,
                "assistant": correction,
            })
        return records

    def _rules_to_records(self, domain: str | None, min_confidence: float, since: datetime | None) -> list[dict]:
        """Convert rules to records."""
        rules = self.store.list_rules(
            min_confidence=min_confidence,
            domain=domain,
        )
        records = []
        for r in rules:
            if since:
                created = r.get("created", "")
                if created and created < since.isoformat():
                    continue

            summary = r.get("summary", "")
            if not summary:
                continue

            records.append({
                "system": f"Rule: {summary}",
                "user": "",
                "assistant": "",
            })
        return records

    def _glossary_to_records(self, domain: str | None) -> list[dict]:
        """Convert glossary terms to Q&A records."""
        if not self.vector_store:
            return []

        records = []
        try:
            terms = self.vector_store.list_glossary_terms()
            for term in terms:
                name = getattr(term, "name", "") or term.get("name", "") if isinstance(term, dict) else term.name
                definition = getattr(term, "definition", "") or (term.get("definition", "") if isinstance(term, dict) else "")
                if not name or not definition:
                    continue

                if domain:
                    term_domain = getattr(term, "domain", "") or (term.get("domain", "") if isinstance(term, dict) else "")
                    if term_domain and term_domain != domain:
                        continue

                records.append({
                    "system": "You are a data analyst assistant.",
                    "user": f"What does '{name}' mean in this context?",
                    "assistant": definition,
                })
        except Exception:
            pass
        return records

    @staticmethod
    def _to_messages(record: dict) -> dict:
        """OpenAI Messages format."""
        messages = []
        if record.get("system"):
            messages.append({"role": "system", "content": record["system"]})
        if record.get("user"):
            messages.append({"role": "user", "content": record["user"]})
        if record.get("assistant"):
            messages.append({"role": "assistant", "content": record["assistant"]})
        return {"messages": messages}

    @staticmethod
    def _to_alpaca(record: dict) -> dict:
        """Alpaca format."""
        instruction = record.get("system", "")
        if record.get("user"):
            instruction = record["user"] if not instruction else f"{instruction}\n\n{record['user']}"
        return {
            "instruction": instruction,
            "input": "",
            "output": record.get("assistant", ""),
        }

    @staticmethod
    def _to_sharegpt(record: dict) -> dict:
        """ShareGPT format."""
        conversations = []
        if record.get("system"):
            conversations.append({"from": "system", "value": record["system"]})
        if record.get("user"):
            conversations.append({"from": "human", "value": record["user"]})
        if record.get("assistant"):
            conversations.append({"from": "gpt", "value": record["assistant"]})
        return {"conversations": conversations}
