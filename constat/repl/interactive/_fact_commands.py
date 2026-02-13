# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Fact commands mixin — facts, remember, forget, learnings, corrections."""

from rich.table import Table

from constat.storage.facts import FactStore
from constat.storage.learnings import LearningStore, LearningCategory, LearningSource


class _FactCommandsMixin:
    """Fact-related REPL commands: facts, remember, forget, learnings, corrections."""

    def _show_facts(self) -> None:
        """Show both persistent and session facts."""
        persistent_facts = self.api.fact_store.list_facts()

        session_facts = {}
        if self.api.session:
            session_facts = self.api.session.fact_resolver.get_all_facts()

        if not persistent_facts and not session_facts:
            self.console.print("[dim]No facts stored.[/dim]")
            self.console.print("[dim]Use /remember to save facts that persist across sessions.[/dim]")
            return

        table = Table(title="Facts", show_header=True, box=None)
        table.add_column("Name", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Description", style="dim")
        table.add_column("Source", style="dim")
        table.add_column("Role", style="blue")

        for name, fact_data in persistent_facts.items():
            value = fact_data.get("value", "")
            desc = fact_data.get("description", "")
            role_id = fact_data.get("role_id") or ""
            table.add_row(name, str(value), desc, "[bold]persistent[/bold]", role_id)

        for name, fact in session_facts.items():
            if name in persistent_facts:
                continue

            desc = fact.description or ""
            role_id = getattr(fact, "role_id", None) or ""
            if fact.source_name:
                source_info = f"{fact.source.value}:{fact.source_name}"
                if fact.api_endpoint:
                    source_info += f" ({fact.api_endpoint})"
            elif fact.api_endpoint:
                endpoint_preview = fact.api_endpoint[:50] + "..." if len(fact.api_endpoint) > 50 else fact.api_endpoint
                source_info = f"api: {endpoint_preview}"
            elif fact.query:
                query_preview = fact.query[:50] + "..." if len(fact.query) > 50 else fact.query
                source_info = f"SQL: {query_preview}"
            elif fact.rule_name:
                source_info = f"rule: {fact.rule_name}"
            elif fact.reasoning and fact.source.value == "user_provided":
                source_info = fact.reasoning[:40] + "..." if len(fact.reasoning) > 40 else fact.reasoning
            else:
                source_info = f"session:{fact.source.value}"
            table.add_row(name, fact.display_value, desc, source_info, role_id)

        self.console.print(table)

        if self.api.session and self.api.session.datastore:
            try:
                facts_df = self.api.session.fact_resolver.get_facts_as_dataframe()
                if not facts_df.empty:
                    self.api.session.datastore.save_dataframe("_facts", facts_df)
                    self.console.print("[dim]Facts synced to _facts table (queryable via SQL)[/dim]")
            except Exception:
                pass

    def _remember_fact(self, fact_text: str) -> None:
        """Remember a fact persistently (survives across sessions).

        Supports two modes:
        1. Promote session fact: /remember <fact-name> [as <new-name>]
        2. Extract from text: /remember my role is CFO
        """
        import re

        if not fact_text.strip():
            self.console.print("[yellow]Usage: /remember <fact>[/yellow]")
            self.console.print("[dim]Examples:[/dim]")
            self.console.print("[dim]  /remember enterprise_churn_rate    - persist a session fact[/dim]")
            self.console.print("[dim]  /remember churn_rate as baseline   - persist with new name[/dim]")
            self.console.print("[dim]  /remember my role is CFO           - extract from text[/dim]")
            return

        session_fact_match = re.match(r'^(\S+)(?:\s+as\s+(\S+))?$', fact_text.strip())

        if session_fact_match and self.api.session:
            fact_name = session_fact_match.group(1)
            new_name = session_fact_match.group(2)

            session_facts = self.api.session.fact_resolver.get_all_facts()

            matching_fact = None
            matching_key = None

            for key, fact in session_facts.items():
                if key == fact_name or key == f"{fact_name}()":
                    matching_fact = fact
                    matching_key = key
                    break
                if fact.name == fact_name:
                    matching_fact = fact
                    matching_key = key
                    break

            if matching_fact:
                persist_name = new_name if new_name else matching_fact.name

                context_parts = [f"Source: {matching_fact.source.value}"]
                if matching_fact.source_name:
                    context_parts.append(f"From: {matching_fact.source_name}")
                if matching_fact.query:
                    context_parts.append(f"Query: {matching_fact.query}")
                if matching_fact.reasoning:
                    context_parts.append(f"Reasoning: {matching_fact.reasoning}")
                if matching_fact.api_endpoint:
                    context_parts.append(f"API: {matching_fact.api_endpoint}")
                if matching_fact.resolved_at:
                    context_parts.append(f"Resolved: {matching_fact.resolved_at.isoformat()}")
                context = "\n".join(context_parts)

                description = matching_fact.description or f"Persisted from session (originally: {matching_key})"

                self.api.fact_store.save_fact(
                    name=persist_name,
                    value=matching_fact.value,
                    description=description,
                    context=context,
                )

                self.console.print(f"[green]Remembered:[/green] {persist_name} = {matching_fact.display_value}")
                self.console.print(f"[dim]Source: {matching_fact.source.value}[/dim]")
                if new_name:
                    self.console.print(f"[dim]Renamed from: {matching_fact.name}[/dim]")
                self.console.print("[dim]This fact will persist across sessions.[/dim]")
                return

        self.display.start_spinner("Extracting fact...")
        try:
            extracted = []
            if self.api.session:
                extracted = self.api.session.fact_resolver.add_user_facts_from_text(fact_text)
            else:
                extracted = self._extract_fact_without_session(fact_text)

            self.display.stop_spinner()

            if extracted:
                for fact in extracted:
                    self.api.fact_store.save_fact(
                        name=fact.name if hasattr(fact, 'name') else fact['name'],
                        value=fact.value if hasattr(fact, 'value') else fact['value'],
                        description=fact.description if hasattr(fact, 'description') else fact.get('description', ''),
                        context=fact.context if hasattr(fact, 'context') else fact.get('context', ''),
                    )
                    name = fact.name if hasattr(fact, 'name') else fact['name']
                    value = fact.value if hasattr(fact, 'value') else fact['value']
                    self.console.print(f"[green]Remembered:[/green] {name} = {value}")
                    self.console.print("[dim]This fact will persist across sessions.[/dim]")
            else:
                if session_fact_match and self.api.session:
                    fact_name = session_fact_match.group(1)
                    self.console.print(f"[yellow]No session fact named '{fact_name}' found.[/yellow]")
                    self.console.print("[dim]Use /facts to see available facts, or provide natural language.[/dim]")
                else:
                    self.console.print("[yellow]Could not extract a fact from that text.[/yellow]")
                    self.console.print("[dim]Try being more explicit, e.g., 'my role is CFO'[/dim]")

        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error:[/red] {e}")

    def _extract_fact_without_session(self, text: str) -> list[dict]:
        """Extract facts from text without an active session (lightweight)."""
        import re

        patterns = [
            (r"my\s+(\w+)\s+is\s+(.+)", lambda m: {"name": f"user_{m.group(1)}", "value": m.group(2).strip(), "description": f"User's {m.group(1)}"}),
            (r"i\s+am\s+(?:a|an)\s+(.+)", lambda m: {"name": "user_role", "value": m.group(1).strip(), "description": "User's role"}),
            (r"(\w+)\s*=\s*(.+)", lambda m: {"name": m.group(1).strip(), "value": m.group(2).strip(), "description": ""}),
            (r"(\w+(?:\s+\w+)?)\s+is\s+(.+)", lambda m: {"name": m.group(1).strip().replace(" ", "_"), "value": m.group(2).strip(), "description": ""}),
        ]

        text_lower = text.lower()
        for pattern, extractor in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return [extractor(match)]

        return []

    def _forget_fact(self, fact_name: str) -> None:
        """Forget a fact by name (checks both persistent and session)."""
        if not fact_name.strip():
            self.console.print("[yellow]Usage: /forget <fact_name>[/yellow]")
            self.console.print("[dim]Use /facts to see fact names[/dim]")
            return

        fact_name = fact_name.strip()
        found = False

        if self.api.fact_store.delete_fact(fact_name):
            self.console.print(f"[green]Forgot persistent fact:[/green] {fact_name}")
            found = True

        if self.api.session:
            facts = self.api.session.fact_resolver.get_all_facts()
            if fact_name in facts:
                self.api.session.fact_resolver._cache.pop(fact_name, None)
                if not found:
                    self.console.print(f"[green]Forgot session fact:[/green] {fact_name}")
                found = True

        if not found:
            self.console.print(f"[yellow]Fact '{fact_name}' not found.[/yellow]")
            self.console.print("[dim]Use /facts to see available facts[/dim]")

    def _handle_correct(self, arg: str) -> None:
        """Handle /correct <text> - explicit user correction."""
        if not arg.strip():
            self.console.print("[yellow]Usage: /correct <correction>[/yellow]")
            self.console.print("[dim]Example: /correct 'active users' means users logged in within 30 days[/dim]")
            return

        self.api.learning_store.save_learning(
            category=LearningCategory.USER_CORRECTION,
            context={
                "previous_question": self.last_problem,
                "correction_text": arg,
            },
            correction=arg,
            source=LearningSource.EXPLICIT_COMMAND,
        )
        self.console.print(f"[green]Learned:[/green] {arg[:60]}{'...' if len(arg) > 60 else ''}")

        self._maybe_auto_compact()

    def _show_learnings(self, arg: str = "") -> None:
        """Handle /learnings [category] - show learnings and rules."""
        category = None
        if arg.strip():
            try:
                category = LearningCategory(arg.strip().lower())
            except ValueError:
                self.console.print(f"[yellow]Unknown category: {arg}[/yellow]")
                self.console.print("[dim]Valid: user_correction, api_error, codegen_error, nl_correction[/dim]")

        rules = self.api.learning_store.list_rules(category=category)
        if rules:
            self.console.print(f"\n[bold]Rules[/bold] ({len(rules)})")
            for r in rules[:10]:
                conf = r.get("confidence", 0) * 100
                applied = r.get("applied_count", 0)
                self.console.print(f"  [{conf:.0f}%] {r['summary'][:60]} [dim](applied {applied}x)[/dim]")

        raw = self.api.learning_store.list_raw_learnings(category=category, limit=20)
        pending = [l for l in raw if not l.get("promoted_to")]
        if pending:
            self.console.print(f"\n[bold]Pending Learnings[/bold] ({len(pending)})")
            for l in pending[:10]:
                cat = l.get("category", "")[:10]
                lid = l.get("id", "")[:12]
                self.console.print(f"  [dim]{lid}[/dim] [{cat}] {l['correction'][:50]}...")

        stats = self.api.learning_store.get_stats()
        if stats.get("total_raw", 0) > 0 or stats.get("total_rules", 0) > 0:
            self.console.print(
                f"\n[dim]Total: {stats.get('unpromoted', 0)} pending, "
                f"{stats.get('total_rules', 0)} rules, "
                f"{stats.get('total_archived', 0)} archived[/dim]"
            )
        else:
            self.console.print("[dim]No learnings yet. Use /correct to add corrections.[/dim]")

    def _compact_learnings(self) -> None:
        """Handle /compact-learnings - trigger compaction."""
        from constat.learning.compactor import LearningCompactor

        if not self.api.session:
            self.console.print("[yellow]Start a session first by asking a question.[/yellow]")
            return

        stats = self.api.learning_store.get_stats()
        if stats.get("unpromoted", 0) < 2:
            self.console.print("[dim]Not enough learnings to compact (need at least 2).[/dim]")
            return

        self.display.start_spinner("Analyzing learnings for patterns...")
        try:
            llm = self.api.session.router._get_provider(self.api.session.router.models["planning"])
            compactor = LearningCompactor(self.api.learning_store, llm)
            result = compactor.compact()

            self.display.stop_spinner()
            self.console.print(f"[green]Compaction complete:[/green]")
            self.console.print(f"  Rules created: {result.rules_created}")
            self.console.print(f"  Learnings archived: {result.learnings_archived}")
            if result.skipped_low_confidence > 0:
                self.console.print(f"  [dim]Skipped (low confidence): {result.skipped_low_confidence}[/dim]")
            if result.errors:
                for err in result.errors:
                    self.console.print(f"  [yellow]Warning: {err}[/yellow]")
        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error:[/red] {e}")

    def _maybe_auto_compact(self) -> None:
        """Check if auto-compaction should trigger and run it."""
        stats = self.api.learning_store.get_stats()
        unpromoted = stats.get("unpromoted", 0)

        result = self.api.maybe_auto_compact()

        if result is not None:
            if result.rules_created > 0:
                self.console.print(
                    f"[green]Created {result.rules_created} rules from "
                    f"{result.learnings_archived} learnings[/green]"
                )
            else:
                self.console.print(
                    f"[dim]No new rules created (reviewed {unpromoted} learnings)[/dim]"
                )

    def _forget_learning(self, learning_id: str) -> None:
        """Handle /forget-learning <id> - delete a learning."""
        learning_id = learning_id.strip()
        if self.api.learning_store.delete_learning(learning_id):
            self.console.print(f"[green]Deleted learning:[/green] {learning_id}")
        else:
            if self.api.learning_store.delete_rule(learning_id):
                self.console.print(f"[green]Deleted rule:[/green] {learning_id}")
            else:
                self.console.print(f"[yellow]Not found:[/yellow] {learning_id}")
                self.console.print("[dim]Use /learnings to see IDs[/dim]")

    def _save_nl_correction(self, correction, full_text: str) -> None:
        """Save an NL-detected correction as a learning."""
        self.api.learning_store.save_learning(
            category=LearningCategory.NL_CORRECTION,
            context={
                "match_type": correction.correction_type,
                "matched_text": correction.matched_text,
                "full_text": full_text,
                "previous_question": self.last_problem,
            },
            correction=full_text,
            source=LearningSource.NL_DETECTION,
        )
        self._maybe_auto_compact()
