# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Prompts mixin: prompt building, source context."""

from __future__ import annotations

import logging

from constat.core.models import TaskType
from constat.session._types import STEP_SYSTEM_PROMPT, STEP_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class PromptsMixin:

    def _is_unclear_input(self, text: str) -> bool:
        """Check if input appears to be unclear, garbage, or a copy-paste error.

        Detects:
        - Terminal prompts (e.g., "(.venv) user@host % command")
        - File paths without context
        - Very short input with no meaningful words
        - Copy-paste errors with shell syntax
        """
        import re

        text = text.strip()

        # Empty or very short without meaningful content
        if len(text) < 3:
            return True

        # Terminal prompt patterns
        terminal_patterns = [
            r'^\(.+\)\s*\(.+\)\s*\w+@\w+',  # (.venv) (base) user@host
            r'^\w+@[\w\-]+\s*[%$#>]',  # user@hostname %
            r'^[%$#>]\s*\w+',  # % command or $ command
            r'^\(.+\)\s*%',  # (.venv) %
            r'constat\s+repl',  # constat repl command
            r'^pip\s+install',  # pip install
            r'^python\s+-',  # python -m
            r'^cd\s+/',  # cd /path
            r'^\s*\$\s*\w+',  # $ command
        ]

        for pattern in terminal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        # Looks like a file path without a question
        if re.match(r'^[/~][\w/\-\.]+$', text) or re.match(r'^[\w]:\\', text):
            return True

        # Contains mostly special characters
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.3 and len(text) > 10:
            return True

        return False

    def _build_step_prompt(self, step) -> str:
        """Build the prompt for generating step code."""
        # Format datastore tables info with column metadata
        if self.datastore:
            tables = self.datastore.list_tables()
            if tables:
                table_lines = ["Available in `store` (load with `store.load_dataframe('name')` or query with SQL):"]
                for t in tables:
                    schema = self.datastore.get_table_schema(t['name'])
                    if schema:
                        col_names = [c['name'] for c in schema]
                        table_lines.append(f"  - {t['name']}: {t['row_count']} rows, columns: {col_names}")
                    else:
                        table_lines.append(f"  - {t['name']}: {t['row_count']} rows")
                datastore_info = "\n".join(table_lines)
            else:
                datastore_info = "(no tables saved yet)"
        else:
            datastore_info = "(no datastore)"

        # Get scratchpad from datastore (persistent) - source of truth for isolation
        if self.datastore:
            scratchpad_context = self.datastore.get_scratchpad_as_markdown()
        else:
            scratchpad_context = self.scratchpad.get_recent_context(max_steps=5)

        # Build source context with semantic search for step-relevant tables
        ctx = self._build_source_context(query=step.goal)

        # Build codegen learnings section - only for code generation steps
        # Skip for summarization, planning, intent classification etc.
        learnings_text = ""
        code_gen_types = {TaskType.PYTHON_ANALYSIS, TaskType.SQL_GENERATION}
        if step.task_type in code_gen_types:
            try:
                learnings_text = self._get_codegen_learnings(step.goal, step.task_type)
            except Exception as e:
                logger.debug(f"Failed to get codegen learnings: {e}")

        # Detect relevant concepts and inject specialized sections
        injected_sections = self._concept_detector.get_sections_for_prompt(
            query=step.goal,
            target="step",
        )

        # Build published artifacts context for name reuse
        published_artifacts_text = ""
        if self.datastore:
            existing_artifacts = self.datastore.list_artifacts()
            named = [a for a in existing_artifacts if a.get("type") not in ("code", "output", "error")]
            if named:
                artifact_lines = ["Published artifacts (reuse these names with viz.save_* to update):"]
                for a in named:
                    artifact_lines.append(f"  - {a['name']} ({a.get('type', 'unknown')})")
                published_artifacts_text = "\n".join(artifact_lines)

        return STEP_PROMPT_TEMPLATE.format(
            system_prompt=STEP_SYSTEM_PROMPT,
            injected_sections=injected_sections,
            schema_overview=ctx["schema_overview"],
            api_overview=ctx["api_overview"],
            domain_context=self._get_system_prompt() or "No additional context.",
            user_facts=ctx["user_facts"],
            learnings=learnings_text,
            datastore_tables=datastore_info,
            published_artifacts=published_artifacts_text,
            scratchpad=scratchpad_context,
            step_number=step.number,
            total_steps=len(self.plan.steps) if self.plan else 1,
            goal=step.goal,
            inputs=", ".join(step.expected_inputs) if step.expected_inputs else "(none)",
            outputs=", ".join(step.expected_outputs) if step.expected_outputs else "(none)",
        )

    def _get_codegen_learnings(self, step_goal: str, task_type: TaskType = None) -> str:
        """Get relevant codegen learnings showing what didn't work vs what did work.

        Args:
            step_goal: The goal of the current step for context matching
            task_type: The task type to filter learnings (SQL vs Python)

        Returns:
            Formatted learnings text for prompt injection
        """
        from constat.storage.learnings import LearningCategory

        if not self.learning_store:
            return ""

        # Determine if this is SQL or Python based on task type
        is_sql = task_type == TaskType.SQL_GENERATION

        lines = []

        # Get rules (compacted learnings) for codegen errors
        rules = self.learning_store.list_rules(
            category=LearningCategory.CODEGEN_ERROR,
            min_confidence=0.6,
        )
        if rules:
            # Filter rules by type - SQL rules mention SQL/query patterns
            sql_keywords = {'sql', 'query', 'select', 'join', 'table', 'column', 'duckdb'}
            filtered_rules = []
            for rule in rules:
                summary_lower = rule.get('summary', '').lower()
                rule_is_sql = any(kw in summary_lower for kw in sql_keywords)
                if is_sql == rule_is_sql:
                    filtered_rules.append(rule)

            if filtered_rules:
                label = "SQL" if is_sql else "Code"
                lines.append(f"\n## {label} Generation Rules (apply these)")
                for rule in filtered_rules[:5]:
                    lines.append(f"- {rule['summary']}")

        # Get recent raw learnings with full context (error vs fix)
        raw_learnings = self.learning_store.list_raw_learnings(
            category=LearningCategory.CODEGEN_ERROR,
            limit=10,
            include_promoted=False,
        )
        if raw_learnings:
            # Filter by type and relevance
            sql_code_patterns = {'SELECT', 'FROM', 'WHERE', 'JOIN', 'INSERT', 'UPDATE', 'store.query'}
            relevant = []
            for l in raw_learnings:
                ctx = l.get("context", {})
                code = ctx.get("original_code", "") + ctx.get("fixed_code", "")
                learning_is_sql = any(pat in code for pat in sql_code_patterns)
                if is_sql == learning_is_sql and self._is_learning_relevant(l, step_goal):
                    relevant.append(l)
            relevant = relevant[:3]  # Limit to 3 detailed examples

            if relevant:
                label = "SQL" if is_sql else "Codegen"
                lines.append(f"\n## Recent {label} Fixes (learn from these)")
                for learning in relevant:
                    ctx = learning.get("context", {})
                    original = ctx.get("original_code", "")
                    fixed = ctx.get("fixed_code", "")
                    error_msg = ctx.get("error_message", "")

                    # Show the contrast
                    lines.append(f"\n### {learning['correction'][:80]}")
                    if error_msg:
                        lines.append(f"**Error:** {error_msg[:100]}")
                    lang = "sql" if is_sql else "python"
                    if original:
                        lines.append(f"**Broken code:**\n```{lang}\n{original[:300]}\n```")
                    if fixed:
                        lines.append(f"**Fixed code:**\n```{lang}\n{fixed[:300]}\n```")

        return "\n".join(lines) if lines else ""

    def _is_learning_relevant(self, learning: dict, step_goal: str) -> bool:
        """Check if a learning is relevant to the current step goal."""
        # Simple keyword overlap check
        goal_words = set(step_goal.lower().split())
        learning_goal = learning.get("context", {}).get("step_goal", "")
        learning_words = set(learning_goal.lower().split())
        correction_words = set(learning.get("correction", "").lower().split())

        # Check for meaningful keyword overlap
        common_words = {"the", "a", "an", "to", "from", "for", "with", "in", "on", "of", "and", "or"}
        goal_keywords = goal_words - common_words
        learning_keywords = (learning_words | correction_words) - common_words

        overlap = goal_keywords & learning_keywords
        return len(overlap) >= 1  # At least one meaningful keyword match

    def _get_system_prompt(self) -> str:
        """Get the system prompt with active role and skills appended.

        Returns:
            The config system prompt + active role prompt + active skills prompts
        """
        parts = []

        # Base system prompt from config
        base_prompt = self.config.system_prompt or ""
        if base_prompt:
            parts.append(base_prompt)

        # Role prompt (if active)
        role_prompt = self.role_manager.get_role_prompt()
        if role_prompt:
            parts.append(role_prompt)

        # Skills prompts (if any active)
        active_skill_objects = self.skill_manager.active_skill_objects
        if active_skill_objects:
            skill_parts = []
            for skill in active_skill_objects:
                skill_dir = self.skill_manager.skills_dir / skill.filename
                scripts_dir = skill_dir / "scripts"
                script_files = []
                if scripts_dir.exists():
                    script_files = sorted(
                        str(f) for f in scripts_dir.iterdir() if f.is_file()
                    )
                if script_files:
                    skill_parts.append(
                        f"## Skill: {skill.name} — EXECUTE, DO NOT REWRITE\n"
                        f"Scripts: {', '.join(script_files)}\n\n"
                        f"**Load the script and call its `run_proof()` function.** Do NOT reimplement the logic.\n\n"
                        f"Generate ONLY this exact pattern — no summary code, no column references, no extra logic:\n"
                        f"```python\n"
                        f"import pandas as pd\n\n"
                        f"# 1. Load script into its own namespace\n"
                        f"_ns = {{}}\n"
                        f"exec(open('{script_files[0]}').read(), _ns)\n\n"
                        f"# 2. Call run_proof() — returns dict[str, str] of Parquet file paths\n"
                        f"file_paths = _ns['run_proof']()\n\n"
                        f"# 3. Load and save each dataset to the session store\n"
                        f"for name, path in file_paths.items():\n"
                        f"    store.save_dataframe(name, pd.read_parquet(path))\n"
                        f"_result = pd.read_parquet(file_paths['_result'])\n"
                        f"```\n\n"
                        f"CRITICAL: Do NOT add print statements that reference specific column names.\n"
                        f"The skill script's output columns may differ from the documentation below.\n"
                        f"Just load, save, and assign `_result`. Nothing else.\n\n"
                        f"Skill documentation (for context only — do NOT hardcode column names from this):\n\n"
                        f"{skill.prompt}"
                    )
                else:
                    skill_parts.append(f"## Skill: {skill.name} (reference)\n{skill.prompt}")
            parts.append("# Active Skills\n\n" + "\n\n".join(skill_parts))

        return "\n\n".join(parts)

    def _build_source_context(self, include_user_facts: bool = True, query: str = None) -> dict:
        """Build context about available data sources (schema, APIs, documents, facts).

        Args:
            include_user_facts: Whether to include resolved user facts
            query: Optional natural language query for semantic source search.
                   When provided, finds relevant tables, documents, and APIs
                   via similarity search and includes targeted context.

        Returns:
            dict with keys: schema_overview, api_overview, doc_overview, user_facts
        """
        # When query is provided, use semantic search across ALL source types
        if query:
            sources = self.find_relevant_sources(query, table_limit=10, doc_limit=5, api_limit=5)
            schema_overview = self._format_relevant_tables(sources.get("tables", []))
            api_overview = self._format_relevant_apis(sources.get("apis", []))
            doc_overview = self._format_relevant_docs(sources.get("documents", []))
        else:
            # Schema overview - prefer preloaded hot tables over full listing
            if self._preloaded_context:
                schema_overview = self._preloaded_context
                schema_overview += "\n\nUse `find_relevant_tables(query)` or `get_table_schema(table)` for other tables."
            else:
                schema_overview = self._get_brief_schema_summary()
                schema_overview += "\n\nUse discovery tools to explore schemas: `find_relevant_tables(query)`, `get_table_schema(table)`"

            # API overview - use self.resources (single source of truth)
            api_overview = ""
            if self.resources.has_apis():
                api_lines = ["\n## Available APIs"]
                for name, api_info in self.resources.apis.items():
                    api_type = api_info.api_type.upper()
                    desc = api_info.description or f"{api_type} endpoint"
                    api_lines.append(f"- **{name}** ({api_type}): {desc}")
                api_overview = "\n".join(api_lines)

            # Document overview - use self.resources (single source of truth)
            doc_overview = ""
            if self.resources.has_documents():
                doc_lines = ["\n## Reference Documents"]
                for name, doc_info in self.resources.documents.items():
                    desc = doc_info.description or doc_info.doc_type
                    doc_lines.append(f"- **{name}**: {desc}")
                doc_overview = "\n".join(doc_lines)

        # User facts
        user_facts = ""
        if include_user_facts:
            try:
                all_facts = self.fact_resolver.get_all_facts()
                if all_facts:
                    fact_lines = ["\n## Known User Facts (use these values in code)"]
                    for name, fact in all_facts.items():
                        fact_lines.append(f"- **{name}**: {fact.value}")
                    user_facts = "\n".join(fact_lines)
            except Exception as e:
                logger.debug(f"Failed to get user facts for context: {e}")

        return {
            "schema_overview": schema_overview,
            "api_overview": api_overview,
            "doc_overview": doc_overview,
            "user_facts": user_facts,
        }

    def _build_available_sources_description(self) -> str:
        """Build a concise description of available data sources for Tier 2 assessment.

        This is used by the fact resolver's Tier 2 LLM assessment to understand
        what sources are available without providing full schema details.
        """
        lines = []

        # Databases (names + descriptions, not full schema)
        if self.config.databases:
            lines.append("Databases:")
            for name, db_config in self.config.databases.items():
                desc = db_config.description or db_config.type or "SQL database"
                lines.append(f"  - {name}: {desc}")

        # Documents
        if self.config.documents:
            lines.append("Documents:")
            for name, doc_config in self.config.documents.items():
                desc = doc_config.description or doc_config.type
                lines.append(f"  - {name}: {desc}")

        # APIs
        if self.config.apis:
            lines.append("APIs:")
            for name, api_config in self.config.apis.items():
                desc = api_config.description or f"{api_config.type} endpoint"
                lines.append(f"  - {name}: {desc}")

        # Config values
        if self.config.system_prompt:
            lines.append("Config: Domain context available in system prompt")

        # Hot tables (preloaded schema)
        if self._preloaded_context:
            lines.append("Hot tables (schema preloaded): See system context")

        # Tool calling documentation
        lines.append("Discovery tools available: find_relevant_tables(), get_table_schema(), search_documents()")

        return "\n".join(lines) if lines else "(no sources configured)"
