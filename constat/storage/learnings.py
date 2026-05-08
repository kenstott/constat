# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Two-tier learning storage for corrections and patterns.

Provides storage for learnings (raw corrections) and rules (compacted patterns)
that persist across sessions, stored in .constat/<user_id>/learnings.yaml.
"""

import re
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml


class LearningCategory(Enum):
    """Categories of learnings.

    Categories:
    - USER_CORRECTION: User manually corrected an output
    - CODEGEN_ERROR: General code generation error (syntax, logic, column names, etc.)
    - EXTERNAL_API_ERROR: Error in code calling external REST/GraphQL APIs
    - HTTP_ERROR: HTTP 4xx/5xx errors from external API calls
    - NL_CORRECTION: Natural language interpretation correction
    - API_ERROR: (deprecated, alias for EXTERNAL_API_ERROR for backward compatibility)
    """
    USER_CORRECTION = "user_correction"
    API_ERROR = "api_error"  # Deprecated: kept for backward compatibility
    EXTERNAL_API_ERROR = "external_api_error"  # Clearer name for API integration errors
    HTTP_ERROR = "http_error"  # 4xx/5xx errors from external APIs
    CODEGEN_ERROR = "codegen_error"
    NL_CORRECTION = "nl_correction"


class LearningSource(Enum):
    """How a learning was captured."""
    AUTO_CAPTURE = "auto_capture"
    EXPLICIT_COMMAND = "explicit_command"
    NL_DETECTION = "nl_detection"


class LearningStore:
    """Two-tier learning storage: raw learnings + compacted rules.

    Storage structure:
    ```yaml
    raw_learnings:
      - id: "learn_001"
        category: "codegen_error"
        created: "2024-01-15T10:30:00Z"
        context: {...}
        correction: "GraphQL API returns data directly"
        source: "auto_capture"
        applied_count: 0
        promoted_to: null

    rules:
      - id: "rule_001"
        category: "api_error"
        summary: "GraphQL APIs return data directly"
        confidence: 0.85
        source_learnings: ["learn_001", "learn_003"]
        tags: ["graphql", "api"]
        applied_count: 12
        created: "2024-01-20T14:00:00Z"

    archive:
      - {...}  # Full learning records preserved after promotion
    ```
    """

    def __init__(self, base_dir: Optional[Path] = None, user_id: str = "default"):
        """Initialize learning store.

        Args:
            base_dir: Base directory for .constat. Defaults to current directory.
            user_id: User ID for user-scoped storage.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(".constat")
        self.user_id = user_id
        self.file_path = self.base_dir / user_id / "learnings.yaml"
        self._data: Optional[dict] = None
        self._lock = threading.Lock()  # Thread-safe file access

    def _load(self) -> dict:
        """Load learnings from YAML file (thread-safe)."""
        if self._data is not None:
            return self._data

        with self._lock:
            # Double-check after acquiring lock
            if self._data is not None:
                return self._data

            if not self.file_path.exists():
                self._data = {"raw_learnings": [], "rules": [], "archive": []}
                return self._data

            with open(self.file_path, "r") as f:
                self._data = yaml.safe_load(f) or {}

            # Ensure all sections exist
            if "raw_learnings" not in self._data:
                self._data["raw_learnings"] = []
            if "rules" not in self._data:
                self._data["rules"] = []
            if "archive" not in self._data:
                self._data["archive"] = []

            return self._data

    def _save(self) -> None:
        """Save learnings to YAML file (thread-safe)."""
        with self._lock:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w") as f:
                yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)

    def _generate_id(self, prefix: str = "learn") -> str:
        """Generate a unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    # -------------------------------------------------------------------------
    # Raw Learnings (Tier 1)
    # -------------------------------------------------------------------------

    def save_learning(
        self,
        category: LearningCategory,
        context: dict,
        correction: str,
        source: LearningSource = LearningSource.AUTO_CAPTURE,
    ) -> str:
        """Save a raw learning.

        Args:
            category: Type of learning (user_correction, api_error, etc.)
            context: Contextual information (error details, code snippets, etc.)
            correction: The learning/correction text
            source: How this learning was captured

        Returns:
            The learning ID
        """
        data = self._load()
        learning_id = self._generate_id("learn")

        learning = {
            "id": learning_id,
            "category": category.value,
            "created": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "correction": correction,
            "source": source.value,
            "applied_count": 0,
            "promoted_to": None,
        }

        data["raw_learnings"].append(learning)
        self._save()
        return learning_id

    def get_learning(self, learning_id: str) -> Optional[dict]:
        """Get a learning by ID.

        Args:
            learning_id: Learning ID

        Returns:
            Learning dict or None if not found
        """
        data = self._load()
        for learning in data["raw_learnings"]:
            if learning["id"] == learning_id:
                return learning.copy()
        # Check archive too
        for learning in data["archive"]:
            if learning["id"] == learning_id:
                return learning.copy()
        return None

    def list_raw_learnings(
        self,
        category: Optional[LearningCategory] = None,
        limit: Optional[int] = 50,
        include_promoted: bool = False,
    ) -> list[dict]:
        """List raw learnings.

        Args:
            category: Filter by category (None for all)
            limit: Maximum number to return
            include_promoted: Include learnings that have been promoted to rules

        Returns:
            List of learning dicts, newest first
        """
        data = self._load()
        learnings = data["raw_learnings"]

        # Filter by category
        if category:
            learnings = [l for l in learnings if l["category"] == category.value]

        # Filter promoted unless requested
        if not include_promoted:
            learnings = [l for l in learnings if not l.get("promoted_to")]

        # Sort by created (newest first) and limit
        learnings = sorted(learnings, key=lambda l: l["created"], reverse=True)
        return [l.copy() for l in learnings[:limit]]

    def delete_learning(self, learning_id: str) -> bool:
        """Delete a learning.

        Args:
            learning_id: Learning ID

        Returns:
            True if deleted, False if not found
        """
        data = self._load()
        for i, learning in enumerate(data["raw_learnings"]):
            if learning["id"] == learning_id:
                del data["raw_learnings"][i]
                self._save()
                return True
        return False

    def increment_applied(self, learning_id: str) -> None:
        """Increment the applied count for a learning."""
        data = self._load()
        for learning in data["raw_learnings"]:
            if learning["id"] == learning_id:
                learning["applied_count"] = learning.get("applied_count", 0) + 1
                self._save()
                return

    # -------------------------------------------------------------------------
    # Rules (Tier 2)
    # -------------------------------------------------------------------------

    def save_rule(
        self,
        summary: str,
        category: LearningCategory,
        confidence: float,
        source_learnings: list[str],
        tags: Optional[list[str]] = None,
    ) -> str:
        """Save a compacted rule.

        Args:
            summary: Rule summary/description
            category: Category of the rule
            confidence: Confidence score (0.0 to 1.0)
            source_learnings: IDs of learnings this rule was derived from
            tags: Optional tags for categorization

        Returns:
            The rule ID
        """
        data = self._load()
        rule_id = self._generate_id("rule")

        rule = {
            "id": rule_id,
            "category": category.value,
            "summary": summary,
            "confidence": confidence,
            "source_learnings": source_learnings,
            "tags": tags or [],
            "applied_count": 0,
            "created": datetime.now(timezone.utc).isoformat(),
        }

        data["rules"].append(rule)
        self._save()
        return rule_id

    def list_rules(
        self,
        category: Optional[LearningCategory] = None,
        min_confidence: float = 0.0,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """List rules.

        Args:
            category: Filter by category (None for all)
            min_confidence: Minimum confidence threshold
            limit: Maximum number of rules to return (None for all)

        Returns:
            List of rule dicts, highest confidence first
        """
        data = self._load()
        rules = data["rules"]

        # Filter by category
        if category:
            rules = [r for r in rules if r["category"] == category.value]

        # Filter by confidence
        rules = [r for r in rules if r.get("confidence", 0) >= min_confidence]

        # Sort by confidence (highest first)
        rules = sorted(rules, key=lambda r: r.get("confidence", 0), reverse=True)

        # Apply limit
        if limit is not None:
            rules = rules[:limit]

        return [r.copy() for r in rules]

    def get_relevant_rules(
        self,
        context: str,
        min_confidence: float = 0.6,
        limit: int = 5,
    ) -> list[dict]:
        """Get rules relevant to the given context.

        Uses simple keyword matching against rule summaries and tags.

        Args:
            context: Context string (problem, step goal, etc.)
            min_confidence: Minimum confidence threshold
            limit: Maximum rules to return

        Returns:
            List of relevant rules, scored by relevance
        """
        rules = self.list_rules(min_confidence=min_confidence)
        if not rules:
            return []

        # Extract keywords from context
        context_lower = context.lower()
        context_words = set(re.findall(r'\w+', context_lower))

        # Score each rule by keyword overlap
        scored_rules = []
        for rule in rules:
            summary_words = set(re.findall(r'\w+', rule["summary"].lower()))
            tag_words = set(t.lower() for t in rule.get("tags", []))
            all_rule_words = summary_words | tag_words

            # Calculate overlap score
            overlap = len(context_words & all_rule_words)
            if overlap > 0:
                score = overlap + rule.get("confidence", 0) + (rule.get("applied_count", 0) * 0.1)
                scored_rules.append((score, rule))

        # Sort by score and return top results
        scored_rules.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_rules[:limit]]

    def increment_rule_applied(self, rule_id: str) -> None:
        """Increment the applied count for a rule."""
        data = self._load()
        for rule in data["rules"]:
            if rule["id"] == rule_id:
                rule["applied_count"] = rule.get("applied_count", 0) + 1
                self._save()
                return

    def update_rule(
        self,
        rule_id: str,
        summary: Optional[str] = None,
        tags: Optional[list[str]] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        """Update an existing rule.

        Args:
            rule_id: Rule ID to update
            summary: New summary (if provided)
            tags: New tags (if provided)
            confidence: New confidence (if provided)

        Returns:
            True if updated, False if rule not found
        """
        data = self._load()
        for rule in data["rules"]:
            if rule["id"] == rule_id:
                if summary is not None:
                    rule["summary"] = summary
                if tags is not None:
                    rule["tags"] = tags
                if confidence is not None:
                    rule["confidence"] = confidence
                rule["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save()
                return True
        return False

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule.

        Args:
            rule_id: Rule ID

        Returns:
            True if deleted, False if not found
        """
        data = self._load()
        for i, rule in enumerate(data["rules"]):
            if rule["id"] == rule_id:
                del data["rules"][i]
                self._save()
                return True
        return False

    # -------------------------------------------------------------------------
    # Archive
    # -------------------------------------------------------------------------

    def archive_learning(self, learning_id: str, rule_id: str) -> bool:
        """Archive a learning after it has been promoted to a rule.

        Args:
            learning_id: Learning ID to archive
            rule_id: Rule ID it was promoted to

        Returns:
            True if archived, False if learning not found
        """
        data = self._load()
        for i, learning in enumerate(data["raw_learnings"]):
            if learning["id"] == learning_id:
                # Mark as promoted
                learning["promoted_to"] = rule_id
                learning["archived_at"] = datetime.now(timezone.utc).isoformat()
                # Move to archive
                data["archive"].append(learning)
                del data["raw_learnings"][i]
                self._save()
                return True
        return False

    def list_archive(self, limit: int = 50) -> list[dict]:
        """List archived learnings.

        Args:
            limit: Maximum number to return

        Returns:
            List of archived learning dicts
        """
        data = self._load()
        archive = sorted(
            data["archive"],
            key=lambda l: l.get("archived_at", l["created"]),
            reverse=True
        )
        return [l.copy() for l in archive[:limit]]

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get learning statistics.

        Returns:
            Dict with counts and breakdown by category
        """
        data = self._load()

        raw = data["raw_learnings"]
        rules = data["rules"]
        archive = data["archive"]

        # Count by category
        raw_by_category = {}
        for learning in raw:
            cat = learning.get("category", "unknown")
            raw_by_category[cat] = raw_by_category.get(cat, 0) + 1

        rules_by_category = {}
        for rule in rules:
            cat = rule.get("category", "unknown")
            rules_by_category[cat] = rules_by_category.get(cat, 0) + 1

        # Count unpromoted
        unpromoted = len([l for l in raw if not l.get("promoted_to")])

        return {
            "total_raw": len(raw),
            "total_rules": len(rules),
            "total_archived": len(archive),
            "unpromoted": unpromoted,
            "raw_by_category": raw_by_category,
            "rules_by_category": rules_by_category,
        }

    def clear_all(self) -> dict:
        """Clear all learnings, rules, and archive.

        Returns:
            Dict with counts of items cleared
        """
        data = self._load()
        counts = {
            "raw_learnings": len(data["raw_learnings"]),
            "rules": len(data["rules"]),
            "archive": len(data["archive"]),
        }
        self._data = {"raw_learnings": [], "rules": [], "archive": []}
        self._save()
        return counts
