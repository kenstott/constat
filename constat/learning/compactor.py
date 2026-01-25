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

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import json
import re

from constat.storage.learnings import LearningStore, LearningCategory

logger = logging.getLogger(__name__)


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    rules_created: int = 0
    rules_strengthened: int = 0  # Existing rules reinforced with new evidence
    rules_merged: int = 0  # Duplicate rules consolidated
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

        Also detects duplicate rules and strengthens existing rules when
        new learnings reinforce them.

        Args:
            dry_run: If True, analyze but don't create rules

        Returns:
            CompactionResult with counts and details
        """
        result = CompactionResult()

        # First, check for and merge duplicate existing rules
        if not dry_run:
            merge_count = self._merge_duplicate_rules()
            result.rules_merged = merge_count

        # Get existing rules for overlap checking
        existing_rules = self.store.list_rules(limit=100)

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

            # Get existing rules for this category
            category_rules = [r for r in existing_rules if r.get("category") == category]

            try:
                # Find groups of similar learnings using LLM
                groups = self._find_similar_groups(learnings)
                result.groups_found += len(groups)

                for group in groups:
                    if len(group) < self.MIN_GROUP_SIZE:
                        logger.debug(f"[compact] Skipping group with {len(group)} learnings (< {self.MIN_GROUP_SIZE})")
                        continue

                    logger.info(f"[compact] Processing group of {len(group)} learnings in {category}")

                    # Check if this group overlaps with an existing rule
                    overlapping_rule = self._find_overlapping_rule(group, category_rules)

                    if overlapping_rule:
                        logger.info(f"[compact] Group overlaps with existing rule: {overlapping_rule['id']}")
                        # Strengthen the existing rule instead of creating new one
                        if not dry_run:
                            strengthened = self._strengthen_rule(overlapping_rule, group)
                            if strengthened:
                                result.rules_strengthened += 1
                                # Archive source learnings under the existing rule
                                for learning in group:
                                    self.store.archive_learning(learning["id"], overlapping_rule["id"])
                                    result.learnings_archived += 1
                                logger.info(f"[compact] Strengthened rule and archived {len(group)} learnings")
                            else:
                                logger.warning(f"[compact] Failed to strengthen rule {overlapping_rule['id']}")
                        continue

                    # No overlap - generate new rule summary
                    logger.info(f"[compact] No overlapping rule, generating new rule summary")
                    rule_data = self._generate_rule_summary(group)
                    if not rule_data:
                        logger.warning(f"[compact] Failed to generate rule summary for group")
                        continue

                    confidence = rule_data.get("confidence", 0)
                    logger.info(f"[compact] Generated rule with confidence {confidence}: {rule_data.get('summary', '')[:80]}")
                    if confidence < self.CONFIDENCE_THRESHOLD:
                        logger.info(f"[compact] Skipping low confidence rule ({confidence} < {self.CONFIDENCE_THRESHOLD})")
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

    def _merge_duplicate_rules(self) -> int:
        """Find and merge duplicate/overlapping rules.

        Returns:
            Number of rules merged
        """
        rules = self.store.list_rules(limit=100)
        if len(rules) < 2:
            return 0

        # Group rules by category
        by_category = defaultdict(list)
        for rule in rules:
            cat = rule.get("category", "unknown")
            by_category[cat].append(rule)

        merged_count = 0

        for category, category_rules in by_category.items():
            if len(category_rules) < 2:
                continue

            # Find duplicate pairs using LLM
            duplicate_groups = self._find_duplicate_rules(category_rules)

            for group in duplicate_groups:
                if len(group) < 2:
                    continue

                # Merge into the first (oldest) rule
                primary = group[0]
                to_merge = group[1:]

                # Generate merged summary
                merged_summary = self._generate_merged_rule_summary(group)
                if merged_summary:
                    # Update primary rule with merged summary
                    self.store.update_rule(
                        rule_id=primary["id"],
                        summary=merged_summary["summary"],
                        tags=merged_summary.get("tags", primary.get("tags", [])),
                    )

                    # Delete duplicate rules
                    for dup_rule in to_merge:
                        self.store.delete_rule(dup_rule["id"])
                        merged_count += 1

        return merged_count

    def _find_duplicate_rules(self, rules: list[dict]) -> list[list[dict]]:
        """Find groups of duplicate/overlapping rules.

        Args:
            rules: List of rule dicts

        Returns:
            List of groups, each containing duplicate rules
        """
        if len(rules) < 2:
            return []

        # Build summaries for LLM
        rule_summaries = []
        for i, r in enumerate(rules[:20]):  # Limit
            rule_summaries.append(f"{i}: {r['summary'][:150]}")

        prompt = f"""Analyze these {len(rule_summaries)} rules and find DUPLICATES or OVERLAPPING rules.

Rules:
{chr(10).join(rule_summaries)}

Two rules are duplicates if they describe the SAME underlying principle, even with different wording.
Return JSON with groups of duplicate rule indices.

Example output:
{{"duplicates": [[0, 3], [1, 5, 7]]}}

Only include groups where rules are genuinely duplicates (same meaning).
Output ONLY valid JSON, no explanation."""

        try:
            response = self.llm.generate(
                system="You are identifying duplicate coding rules.",
                user_message=prompt,
                max_tokens=300,
            )

            content = response.strip() if isinstance(response, str) else response.content.strip()
            if "```" in content:
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    content = match.group(1)

            data = json.loads(content)
            duplicate_indices = data.get("duplicates", [])

            # Convert indices to rules
            groups = []
            for indices in duplicate_indices:
                group = [rules[i] for i in indices if i < len(rules)]
                if len(group) >= 2:
                    groups.append(group)

            return groups

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Duplicate detection failed: {e}")
            return []

    def _find_overlapping_rule(self, learnings: list[dict], rules: list[dict]) -> Optional[dict]:
        """Check if learnings overlap with an existing rule.

        Args:
            learnings: Group of similar learnings
            rules: Existing rules in the same category

        Returns:
            Overlapping rule dict, or None if no overlap
        """
        if not rules:
            return None

        # Build context for LLM
        learning_summaries = [l["correction"][:100] for l in learnings[:5]]
        rule_summaries = [(i, r["summary"][:150]) for i, r in enumerate(rules[:10])]

        prompt = f"""Do these learnings overlap with any existing rule?

Learnings (corrections from errors):
{chr(10).join(f'- {s}' for s in learning_summaries)}

Existing Rules:
{chr(10).join(f'{i}: {s}' for i, s in rule_summaries)}

If the learnings describe the SAME pattern as an existing rule, return that rule's index.
If no overlap, return -1.

Output JSON: {{"overlapping_rule_index": <index or -1>}}
Output ONLY valid JSON."""

        try:
            response = self.llm.generate(
                system="You are checking if learnings match existing rules.",
                user_message=prompt,
                max_tokens=50,
            )

            content = response.strip() if isinstance(response, str) else response.content.strip()
            if "```" in content:
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    content = match.group(1)

            data = json.loads(content)
            idx = data.get("overlapping_rule_index", -1)

            if idx >= 0 and idx < len(rules):
                logger.info(f"[find_overlap] Found overlapping rule at index {idx}: {rules[idx]['id']}")
                return rules[idx]
            else:
                logger.debug(f"[find_overlap] No overlapping rule found (idx={idx})")

        except Exception as e:
            logger.warning(f"[find_overlap] Error checking overlap: {e}")

        return None

    def _strengthen_rule(self, rule: dict, learnings: list[dict]) -> bool:
        """Strengthen an existing rule with new evidence from learnings.

        Makes the rule more emphatic and/or adds brief examples.

        Args:
            rule: Existing rule to strengthen
            learnings: New learnings that reinforce this rule

        Returns:
            True if rule was updated
        """
        learning_summaries = [l["correction"][:100] for l in learnings[:5]]
        contexts = []
        for l in learnings[:3]:
            ctx = l.get("context", {})
            if isinstance(ctx, dict):
                err = ctx.get("error_message", "")[:80]
                if err:
                    contexts.append(err)

        prompt = f"""Strengthen this rule based on new evidence.

Current rule:
{rule['summary']}

New learnings that reinforce this:
{chr(10).join(f'- {s}' for s in learning_summaries)}

Error contexts:
{chr(10).join(f'- {c}' for c in contexts) if contexts else 'N/A'}

Create an improved rule that is:
1. More emphatic (use "ALWAYS", "NEVER", "CRITICAL" where appropriate)
2. Includes a brief inline example if helpful (e.g., "...use df.columns to check first")
3. Still concise (1-2 sentences max)

Output JSON: {{"strengthened_summary": "...", "tags": [...]}}
Output ONLY valid JSON."""

        try:
            response = self.llm.generate(
                system="You are improving coding rules to be clearer and more actionable.",
                user_message=prompt,
                max_tokens=300,
            )

            content = response.strip() if isinstance(response, str) else response.content.strip()
            if "```" in content:
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    content = match.group(1)

            data = json.loads(content)
            new_summary = data.get("strengthened_summary")
            new_tags = data.get("tags", rule.get("tags", []))

            if new_summary and new_summary != rule["summary"]:
                self.store.update_rule(
                    rule_id=rule["id"],
                    summary=new_summary,
                    tags=new_tags,
                )
                logger.info(f"[strengthen] Updated rule {rule['id']} with new summary")
                return True
            else:
                logger.warning(f"[strengthen] No change to rule - summary unchanged or empty")

        except Exception as e:
            logger.error(f"[strengthen] Error strengthening rule: {e}")

        return False

    def _generate_merged_rule_summary(self, rules: list[dict]) -> Optional[dict]:
        """Generate a merged summary from duplicate rules.

        Args:
            rules: List of duplicate rules to merge

        Returns:
            Dict with merged summary and tags, or None on failure
        """
        summaries = [r["summary"] for r in rules]
        all_tags = set()
        for r in rules:
            all_tags.update(r.get("tags", []))

        prompt = f"""Merge these duplicate rules into ONE clear, comprehensive rule.

Rules to merge:
{chr(10).join(f'- {s}' for s in summaries)}

Create a single rule that:
1. Captures the essence of all duplicates
2. Uses emphatic language (ALWAYS, NEVER, CRITICAL)
3. Is concise but complete (1-2 sentences)

Output JSON: {{"summary": "...", "tags": [...]}}
Output ONLY valid JSON."""

        try:
            response = self.llm.generate(
                system="You are consolidating duplicate coding rules.",
                user_message=prompt,
                max_tokens=200,
            )

            content = response.strip() if isinstance(response, str) else response.content.strip()
            if "```" in content:
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    content = match.group(1)

            data = json.loads(content)
            # Combine tags from LLM with existing tags
            data["tags"] = list(set(data.get("tags", [])) | all_tags)
            return data

        except Exception:
            return None

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

            result = json.loads(content)
            logger.debug(f"[generate_rule] Generated: {result}")
            return result

        except Exception as e:
            logger.error(f"[generate_rule] Error generating rule summary: {e}")
            return None
