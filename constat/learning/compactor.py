# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Learning compactor for promoting raw learnings to rules.

Analyzes patterns in raw learnings and creates generalized rules
when sufficient similar learnings accumulate.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import json
import re

from constat.storage.learnings import LearningStore, LearningCategory


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    rules_created: int = 0
    learnings_archived: int = 0
    learnings_expired: int = 0  # Stale learnings that never grouped
    groups_found: int = 0
    skipped_low_confidence: int = 0
    errors: list[str] = field(default_factory=list)


class LearningCompactor:
    """Compacts raw learnings into rules using LLM pattern detection.

    Configuration:
    - CONFIDENCE_THRESHOLD: Minimum confidence for rule creation (default 0.60)
    - AUTO_COMPACT_THRESHOLD: Trigger auto-compact when unpromoted learnings exceed this (default 50)
    - MIN_GROUP_SIZE: Minimum learnings needed to form a rule (default 2)
    """

    CONFIDENCE_THRESHOLD = 0.60
    AUTO_COMPACT_THRESHOLD = 50
    MIN_GROUP_SIZE = 2

    def __init__(self, learning_store: LearningStore, llm):
        """Initialize compactor.

        Args:
            learning_store: LearningStore instance
            llm: LLM provider (BaseLLMProvider) for pattern analysis
        """
        self.store = learning_store
        self.llm = llm

    def should_auto_compact(self) -> bool:
        """Check if auto-compaction should trigger.

        Returns:
            True if unpromoted learnings exceed threshold
        """
        stats = self.store.get_stats()
        return stats.get("unpromoted", 0) >= self.AUTO_COMPACT_THRESHOLD

    def compact(self, dry_run: bool = False) -> CompactionResult:
        """Run compaction: group similar learnings and promote to rules.

        Args:
            dry_run: If True, analyze but don't create rules

        Returns:
            CompactionResult with counts and details
        """
        result = CompactionResult()

        # Get unpromoted learnings
        all_learnings = self.store.list_raw_learnings(limit=200, include_promoted=False)
        if len(all_learnings) < self.MIN_GROUP_SIZE:
            return result

        # Group by category first
        by_category = defaultdict(list)
        for learning in all_learnings:
            cat = learning.get("category", "unknown")
            by_category[cat].append(learning)

        # Process each category
        for category, learnings in by_category.items():
            if len(learnings) < self.MIN_GROUP_SIZE:
                continue

            try:
                # Find groups of similar learnings using LLM
                groups = self._find_similar_groups(learnings)
                result.groups_found += len(groups)

                for group in groups:
                    if len(group) < self.MIN_GROUP_SIZE:
                        continue

                    # Generate rule summary
                    rule_data = self._generate_rule_summary(group)
                    if not rule_data:
                        continue

                    confidence = rule_data.get("confidence", 0)
                    if confidence < self.CONFIDENCE_THRESHOLD:
                        result.skipped_low_confidence += 1
                        continue

                    if not dry_run:
                        # Create rule
                        rule_id = self.store.save_rule(
                            summary=rule_data["summary"],
                            category=LearningCategory(category),
                            confidence=confidence,
                            source_learnings=[l["id"] for l in group],
                            tags=rule_data.get("tags", []),
                        )

                        # Archive source learnings
                        for learning in group:
                            self.store.archive_learning(learning["id"], rule_id)
                            result.learnings_archived += 1

                        result.rules_created += 1

            except Exception as e:
                result.errors.append(f"Error processing {category}: {str(e)}")

        return result

    def _find_similar_groups(self, learnings: list[dict]) -> list[list[dict]]:
        """Group similar learnings using LLM analysis.

        Args:
            learnings: List of learning dicts

        Returns:
            List of groups, where each group is a list of similar learnings
        """
        if len(learnings) <= 3:
            # Small number - check if they're all similar
            if self._are_similar(learnings):
                return [learnings]
            return []

        # Build summary of learnings for LLM
        learning_summaries = []
        for i, l in enumerate(learnings[:30]):  # Limit to avoid token limits
            summary = f"{i}: {l['correction'][:100]}"
            learning_summaries.append(summary)

        prompt = f"""Analyze these {len(learning_summaries)} learnings and group similar ones.

Learnings:
{chr(10).join(learning_summaries)}

Group learnings that describe the SAME underlying pattern or rule.
Return JSON array of groups, where each group is an array of learning indices.

Example output:
{{"groups": [[0, 3, 7], [1, 4], [2, 5, 6, 8]]}}

Only include groups with 2+ learnings that share a clear common pattern.
Output ONLY valid JSON, no explanation."""

        try:
            response = self.llm.generate(
                system="You are analyzing code learnings to find common patterns.",
                user_message=prompt,
                max_tokens=500,
            )

            # Parse response (generate returns string, not object)
            content = response.strip() if isinstance(response, str) else response.content.strip()
            # Extract JSON if wrapped in markdown
            if "```" in content:
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    content = match.group(1)

            data = json.loads(content)
            groups_indices = data.get("groups", [])

            # Convert indices to actual learnings
            groups = []
            for indices in groups_indices:
                group = [learnings[i] for i in indices if i < len(learnings)]
                if len(group) >= self.MIN_GROUP_SIZE:
                    groups.append(group)

            return groups

        except Exception as e:
            # Fallback: simple keyword-based grouping
            import logging
            logging.getLogger(__name__).warning(f"LLM grouping failed: {e}, using keyword fallback")
            return self._keyword_grouping(learnings)

    def _keyword_grouping(self, learnings: list[dict]) -> list[list[dict]]:
        """Simple keyword-based grouping fallback.

        Groups learnings that share significant keywords.
        """
        # Extract keywords from each learning
        keyword_to_learnings = defaultdict(list)
        for learning in learnings:
            text = learning.get("correction", "").lower()
            # Extract significant words (4+ chars, not common words)
            words = re.findall(r'\b[a-z]{4,}\b', text)
            common = {"that", "this", "with", "from", "have", "been", "when", "should", "could", "would"}
            keywords = [w for w in words if w not in common]

            for kw in set(keywords):
                keyword_to_learnings[kw].append(learning)

        # Find keywords that group multiple learnings
        groups = []
        used = set()

        for kw, kw_learnings in sorted(keyword_to_learnings.items(), key=lambda x: -len(x[1])):
            # Filter to unused learnings
            group = [l for l in kw_learnings if l["id"] not in used]

            if len(group) >= self.MIN_GROUP_SIZE:
                groups.append(group)
                for l in group:
                    used.add(l["id"])

        return groups

    def _are_similar(self, learnings: list[dict]) -> bool:
        """Quick check if learnings are similar using LLM."""
        if len(learnings) < 2:
            return False

        corrections = [l["correction"][:100] for l in learnings]
        prompt = f"""Are these learnings describing the same pattern? Answer YES or NO.

{chr(10).join(f'- {c}' for c in corrections)}"""

        try:
            response = self.llm.generate(
                system="You are checking if code learnings describe the same pattern.",
                user_message=prompt,
                max_tokens=10,
            )
            # generate() returns string directly
            return "yes" in response.lower()
        except Exception:
            return False

    def _generate_rule_summary(self, group: list[dict]) -> Optional[dict]:
        """Generate a rule summary from a group of similar learnings.

        Args:
            group: List of similar learning dicts

        Returns:
            Dict with summary, confidence, tags or None on failure
        """
        corrections = [l["correction"] for l in group]
        contexts = []
        for l in group[:3]:  # Sample contexts
            ctx = l.get("context", {})
            if isinstance(ctx, dict):
                contexts.append(str(ctx)[:200])

        prompt = f"""Create a single rule from these {len(group)} similar learnings.

Learnings:
{chr(10).join(f'- {c}' for c in corrections)}

Context examples:
{chr(10).join(contexts) if contexts else 'N/A'}

Output JSON with:
- summary: A clear, actionable rule (1-2 sentences)
- confidence: How confident you are this is a valid pattern (0.0-1.0)
- tags: 2-4 relevant keywords for finding this rule later

Example:
{{"summary": "Always cast date columns to datetime before comparison in SQL queries", "confidence": 0.85, "tags": ["datetime", "sql", "cast"]}}

Output ONLY valid JSON, no explanation."""

        try:
            response = self.llm.generate(
                system="You are creating coding best practices from observed patterns.",
                user_message=prompt,
                max_tokens=300,
            )

            # generate() returns string directly
            content = response.strip()
            # Extract JSON if wrapped in markdown
            if "```" in content:
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    content = match.group(1)

            return json.loads(content)

        except Exception:
            return None
