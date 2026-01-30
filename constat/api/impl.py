# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""ConstatAPIImpl - Implementation of the ConstatAPI protocol.

Wraps Session and delegates to detection/summarization/learning modules.
Converts Session dicts to frozen dataclasses.
"""

from datetime import datetime, timezone
from typing import Any, Callable, Optional

from constat.context import ContextEstimator, ContextCompactor
from constat.context import ContextStats as ContextStatsRaw
from constat.context import CompactionResult as CompactionResultRaw
from constat.api.detection import detect_display_overrides, detect_nl_correction
from constat.api.learning import maybe_auto_compact
from constat.api.summarization import (
    summarize_facts,
    summarize_plan,
    summarize_session,
    summarize_table,
)
from constat.api.types import (
    ArtifactInfo,
    ContextCompactionResult,
    ContextStats,
    CorrectionDetection,
    DisplayOverrides,
    Fact,
    FollowUpResult,
    Learning,
    LearningCompactionResult,
    ReplayResult,
    ResumeResult,
    Rule,
    SavedPlan,
    SessionState,
    SolveResult,
    StepInfo,
    SummarizeResult,
)
from constat.execution.mode import PlanApprovalRequest, PlanApprovalResponse
from constat.learning.compactor import LearningCompactor
from constat.session import Session
from constat.storage.facts import FactStore
from constat.storage.learnings import LearningCategory, LearningSource, LearningStore


# Type alias for event callbacks
EventCallback = Callable[[str, dict[str, Any]], None]


class ConstatAPIImpl:
    """Implementation of the ConstatAPI protocol.

    Wraps a Session instance and provides a clean API boundary
    for REPL/UI consumers.
    """

    def __init__(
        self,
        session: Session,
        fact_store: FactStore,
        learning_store: LearningStore,
    ):
        """Initialize the API implementation.

        Args:
            session: Underlying Session instance
            fact_store: Persistent fact storage
            learning_store: Learning/correction storage
        """
        self._session = session
        self._fact_store = fact_store
        self._learning_store = learning_store
        self._event_callbacks: list[EventCallback] = []

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    def solve(
        self,
        problem: str,
        *,
        require_approval: bool = True,
    ) -> SolveResult:
        """Solve a new problem, generating and executing a plan."""
        # Store original approval setting and temporarily override
        original_require = self._session.session_config.require_approval
        self._session.session_config.require_approval = require_approval

        try:
            result = self._session.solve(problem)
            return self._convert_solve_result(result)
        except Exception as e:
            return SolveResult(success=False, error=str(e))
        finally:
            self._session.session_config.require_approval = original_require

    def follow_up(self, question: str) -> FollowUpResult:
        """Ask a follow-up question in the current session context."""
        try:
            result = self._session.follow_up(question)
            return self._convert_follow_up_result(result)
        except Exception as e:
            return FollowUpResult(success=False, error=str(e))

    def resume(self, session_id: str) -> ResumeResult:
        """Resume a previous session."""
        try:
            success = self._session.resume(session_id)
            if success:
                plan = self._session.plan
                completed = len(getattr(self._session, 'step_results', []))
                return ResumeResult(
                    success=True,
                    session_id=session_id,
                    plan_goal=plan.goal if plan else None,
                    completed_steps=completed,
                    total_steps=len(plan.steps) if plan else 0,
                )
            else:
                return ResumeResult(
                    success=False,
                    error=f"Session '{session_id}' not found",
                )
        except Exception as e:
            return ResumeResult(success=False, error=str(e))

    def replay(self, plan_id: str) -> ReplayResult:
        """Replay a saved plan."""
        try:
            # Get saved plan from history
            plan_detail = self._session.history.get_saved_plan(plan_id)
            if not plan_detail:
                return ReplayResult(
                    success=False,
                    error=f"Plan '{plan_id}' not found",
                )

            # Execute the plan
            result = self._session.solve(plan_detail.problem)
            artifacts = self._extract_artifacts(result)

            return ReplayResult(
                success=result.get("success", False),
                steps_executed=len(result.get("step_results", [])),
                total_steps=len(result.get("plan", {}).get("steps", [])),
                artifacts=artifacts,
                error=result.get("error"),
            )
        except Exception as e:
            return ReplayResult(success=False, error=str(e))

    # -------------------------------------------------------------------------
    # State and Context
    # -------------------------------------------------------------------------

    def get_state(self) -> SessionState:
        """Get current session state."""
        session = self._session
        plan = session.plan
        step_results = getattr(session, 'step_results', [])

        tables: tuple[str, ...] = ()
        if session.datastore:
            table_list = session.datastore.list_tables()
            tables = tuple(t['name'] for t in table_list)

        facts_count = 0
        if session.fact_resolver:
            facts_count = len(session.fact_resolver.get_all_facts())

        mode = None
        if hasattr(session, 'current_mode'):
            mode = str(session.current_mode)

        return SessionState(
            session_id=session.session_id,
            has_plan=plan is not None,
            plan_goal=plan.goal if plan else None,
            plan_steps=len(plan.steps) if plan else 0,
            completed_steps=len(step_results),
            tables=tables,
            facts_count=facts_count,
            mode=mode,
        )

    def get_context_stats(self) -> ContextStats:
        """Get context token usage statistics."""
        if not self._session.datastore:
            return ContextStats(
                total_tokens=0,
                scratchpad_tokens=0,
                state_tokens=0,
                table_metadata_tokens=0,
                artifact_tokens=0,
                scratchpad_entries=0,
                state_variables=0,
                tables=0,
                artifacts=0,
                is_warning=False,
                is_critical=False,
            )

        estimator = ContextEstimator(self._session.datastore)
        raw_stats = estimator.estimate()

        return ContextStats(
            total_tokens=raw_stats.total_tokens,
            scratchpad_tokens=raw_stats.scratchpad_tokens,
            state_tokens=raw_stats.state_tokens,
            table_metadata_tokens=raw_stats.table_metadata_tokens,
            artifact_tokens=raw_stats.artifact_tokens,
            scratchpad_entries=raw_stats.scratchpad_entries,
            state_variables=raw_stats.state_variables,
            tables=raw_stats.tables,
            artifacts=raw_stats.artifacts,
            is_warning=raw_stats.is_warning,
            is_critical=raw_stats.is_critical,
        )

    def compact_context(self) -> ContextCompactionResult:
        """Compact session context to reduce token usage."""
        if not self._session.datastore:
            return ContextCompactionResult(
                original_tokens=0,
                compacted_tokens=0,
                tokens_saved=0,
                entries_removed=0,
                tables_sampled=0,
            )

        # Get pre-compaction stats
        estimator = ContextEstimator(self._session.datastore)
        pre_stats = estimator.estimate()

        # Run compaction
        compactor = ContextCompactor(self._session.datastore, self._session.llm)
        raw_result = compactor.compact()

        return ContextCompactionResult(
            original_tokens=raw_result.tokens_before,
            compacted_tokens=raw_result.tokens_after,
            tokens_saved=raw_result.tokens_saved,
            entries_removed=raw_result.scratchpad_entries_summarized,
            tables_sampled=raw_result.tables_sampled,
        )

    def reset_context(self) -> None:
        """Reset session context, clearing plan and datastore."""
        self._session.plan = None
        self._session.scratchpad.clear()
        if self._session.datastore:
            # Clear tables but keep datastore
            for table in self._session.datastore.list_tables():
                self._session.datastore.drop_table(table['name'])

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_approval_callback(
        self,
        callback: Callable[[PlanApprovalRequest], PlanApprovalResponse],
    ) -> None:
        """Set callback for plan approval requests."""
        self._session.set_approval_callback(callback)

    def on_event(self, callback: EventCallback) -> None:
        """Register callback for session events."""
        self._event_callbacks.append(callback)

        # Also register with session for step events
        def session_event_handler(event):
            callback(event.event_type, event.data)

        self._session.add_event_handler(session_event_handler)

    def set_clarification_callback(self, callback: Callable) -> None:
        """Set callback for clarification requests."""
        self._session.set_clarification_callback(callback)

    # -------------------------------------------------------------------------
    # Low-Level Access (for transitional use)
    # -------------------------------------------------------------------------

    @property
    def session(self) -> Session:
        """Access the underlying Session for operations not yet in API.

        Note: This is for transitional use. Prefer API methods when available.
        """
        return self._session

    @property
    def fact_store(self) -> FactStore:
        """Access the fact store."""
        return self._fact_store

    @property
    def learning_store(self) -> LearningStore:
        """Access the learning store."""
        return self._learning_store

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def cancel_execution(self) -> None:
        """Cancel the current execution."""
        self._session.cancel_execution()

    def save_plan(
        self,
        name: str,
        problem: str,
        *,
        user_id: str = "default",
        shared: bool = False,
    ) -> None:
        """Save the current plan."""
        self._session.save_plan(name, problem, user_id=user_id, shared=shared)

    def add_database(
        self,
        name: str,
        db_type: str,
        uri: str,
        description: str = "",
    ) -> None:
        """Add a database to the session."""
        self._session.add_database(name, db_type, uri, description)

    def add_file(
        self,
        name: str,
        uri: str,
        auth: str = "",
        description: str = "",
    ) -> None:
        """Add a file to the session."""
        self._session.add_file(name, uri, auth, description)

    def get_all_databases(self) -> dict:
        """Get all configured databases."""
        return self._session.get_all_databases()

    def get_all_files(self) -> dict:
        """Get all configured files."""
        return self._session.get_all_files()

    def refresh_metadata(self) -> dict:
        """Refresh database metadata."""
        return self._session.refresh_metadata()

    def audit(self) -> dict:
        """Run an audit of the session."""
        return self._session.audit()

    def prove_conversation(self) -> dict:
        """Generate proof tree for the conversation."""
        return self._session.prove_conversation()

    def maybe_auto_compact(self) -> Optional[LearningCompactionResult]:
        """Auto-compact learnings if threshold reached."""
        result = maybe_auto_compact(self._session, self._learning_store)
        if result is None:
            return None
        return LearningCompactionResult(
            rules_created=result.rules_created,
            rules_strengthened=result.rules_strengthened,
            rules_merged=result.rules_merged,
            learnings_archived=result.learnings_archived,
            learnings_expired=result.learnings_expired,
            groups_found=result.groups_found,
            skipped_low_confidence=result.skipped_low_confidence,
            errors=tuple(result.errors),
        )

    # -------------------------------------------------------------------------
    # Facts
    # -------------------------------------------------------------------------

    def get_facts(self) -> dict[str, Fact]:
        """Get all persistent facts."""
        raw_facts = self._fact_store.list_facts()
        result = {}

        for name, data in raw_facts.items():
            created_str = data.get("created", "")
            try:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created = datetime.now(timezone.utc)

            result[name] = Fact(
                name=name,
                value=data.get("value"),
                description=data.get("description", ""),
                context=data.get("context", ""),
                created=created,
            )

        return result

    def remember_fact(
        self,
        name: str,
        value: Any,
        description: str = "",
        context: str = "",
    ) -> Fact:
        """Remember a persistent fact."""
        self._fact_store.save_fact(
            name=name,
            value=value,
            description=description,
            context=context,
        )

        # Also add to session fact resolver if available
        if self._session.fact_resolver:
            self._session.fact_resolver.add_user_fact(
                fact_name=name,
                value=value,
                reasoning="Added via API",
                description=description,
            )

        return Fact(
            name=name,
            value=value,
            description=description,
            context=context,
            created=datetime.now(timezone.utc),
        )

    def forget_fact(self, name: str) -> bool:
        """Forget a persistent fact."""
        return self._fact_store.delete_fact(name)

    def extract_facts_from_text(self, text: str) -> list[dict[str, Any]]:
        """Extract potential facts from natural language text."""
        if not self._session.fact_resolver:
            return []

        return self._session.fact_resolver.add_user_facts_from_text(text)

    # -------------------------------------------------------------------------
    # Learnings
    # -------------------------------------------------------------------------

    def get_learnings(
        self,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> list[Learning]:
        """Get raw learnings."""
        cat = LearningCategory(category) if category else None
        raw_learnings = self._learning_store.list_raw_learnings(
            category=cat,
            limit=limit,
        )

        result = []
        for data in raw_learnings:
            created_str = data.get("created", "")
            try:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created = datetime.now(timezone.utc)

            result.append(Learning(
                id=data.get("id", ""),
                category=data.get("category", ""),
                correction=data.get("correction", ""),
                context=data.get("context", {}),
                source=data.get("source", ""),
                created=created,
                applied_count=data.get("applied_count", 0),
                promoted_to=data.get("promoted_to"),
            ))

        return result

    def get_rules(
        self,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> list[Rule]:
        """Get compacted rules."""
        cat = LearningCategory(category) if category else None
        raw_rules = self._learning_store.list_rules(
            category=cat,
            min_confidence=min_confidence,
        )

        result = []
        for data in raw_rules:
            created_str = data.get("created", "")
            try:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created = datetime.now(timezone.utc)

            result.append(Rule(
                id=data.get("id", ""),
                category=data.get("category", ""),
                summary=data.get("summary", ""),
                confidence=data.get("confidence", 0.0),
                source_learnings=tuple(data.get("source_learnings", [])),
                tags=tuple(data.get("tags", [])),
                created=created,
                applied_count=data.get("applied_count", 0),
            ))

        return result

    def save_correction(
        self,
        category: str,
        context: dict[str, Any],
        correction: str,
    ) -> str:
        """Save a correction as a learning."""
        cat = LearningCategory(category)
        return self._learning_store.save_learning(
            category=cat,
            context=context,
            correction=correction,
            source=LearningSource.EXPLICIT_COMMAND,
        )

    def compact_learnings(self, dry_run: bool = False) -> LearningCompactionResult:
        """Compact learnings into rules."""
        # Get LLM for compaction
        model_spec = self._session.router.routing_config.get_models_for_task("general")[0]
        llm = self._session.router._get_provider(model_spec)

        compactor = LearningCompactor(self._learning_store, llm)
        result = compactor.compact(dry_run=dry_run)

        return LearningCompactionResult(
            rules_created=result.rules_created,
            rules_strengthened=result.rules_strengthened,
            rules_merged=result.rules_merged,
            learnings_archived=result.learnings_archived,
            learnings_expired=result.learnings_expired,
            groups_found=result.groups_found,
            skipped_low_confidence=result.skipped_low_confidence,
            errors=tuple(result.errors),
        )

    def forget_learning(self, learning_id: str) -> bool:
        """Delete a learning."""
        return self._learning_store.delete_learning(learning_id)

    # -------------------------------------------------------------------------
    # Detection
    # -------------------------------------------------------------------------

    def detect_display_overrides(self, text: str) -> DisplayOverrides:
        """Detect display preference overrides in natural language."""
        return detect_display_overrides(text)

    def detect_nl_correction(self, text: str) -> CorrectionDetection:
        """Detect if text contains a correction pattern."""
        return detect_nl_correction(text)

    def save_nl_correction(
        self,
        original_text: str,
        correction_type: str,
        matched_text: str,
    ) -> str:
        """Save a detected natural language correction."""
        return self._learning_store.save_learning(
            category=LearningCategory.NL_CORRECTION,
            context={
                "original_text": original_text,
                "correction_type": correction_type,
                "matched_text": matched_text,
            },
            correction=original_text,
            source=LearningSource.NL_DETECTION,
        )

    # -------------------------------------------------------------------------
    # Summarization
    # -------------------------------------------------------------------------

    def summarize_plan(self) -> SummarizeResult:
        """Summarize the current execution plan."""
        return summarize_plan(self._session, self._session.llm)

    def summarize_session(self) -> SummarizeResult:
        """Summarize the current session state."""
        return summarize_session(self._session, self._session.llm)

    def summarize_facts(self) -> SummarizeResult:
        """Summarize all cached facts."""
        return summarize_facts(self._session, self._session.llm)

    def summarize_table(self, table_name: str) -> SummarizeResult:
        """Summarize a specific table's contents."""
        return summarize_table(self._session, table_name, self._session.llm)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _convert_solve_result(self, result: dict) -> SolveResult:
        """Convert Session.solve() dict to SolveResult."""
        success = result.get("success", False)

        plan = result.get("plan", {})
        plan_goal = plan.get("goal") if isinstance(plan, dict) else None

        steps = self._extract_steps(result)
        artifacts = self._extract_artifacts(result)
        tables = self._extract_tables(result)
        suggestions = tuple(result.get("suggestions", []))

        return SolveResult(
            success=success,
            answer=result.get("final_answer") or result.get("summary") or result.get("answer"),
            plan_goal=plan_goal,
            steps=steps,
            artifacts=artifacts,
            tables_created=tables,
            suggestions=suggestions,
            error=result.get("error"),
            raw_output=result.get("raw_output"),
        )

    def _convert_follow_up_result(self, result: dict) -> FollowUpResult:
        """Convert Session.follow_up() dict to FollowUpResult."""
        success = result.get("success", False)

        steps = self._extract_steps(result)
        artifacts = self._extract_artifacts(result)
        tables = self._extract_tables(result)
        suggestions = tuple(result.get("suggestions", []))

        return FollowUpResult(
            success=success,
            answer=result.get("final_answer") or result.get("summary") or result.get("answer"),
            steps=steps,
            artifacts=artifacts,
            tables_created=tables,
            suggestions=suggestions,
            error=result.get("error"),
            raw_output=result.get("raw_output"),
        )

    def _extract_steps(self, result: dict) -> tuple[StepInfo, ...]:
        """Extract step information from result dict."""
        steps = []
        plan = result.get("plan", {})
        plan_steps = plan.get("steps", []) if isinstance(plan, dict) else []
        step_results = result.get("step_results", [])

        for i, step in enumerate(plan_steps):
            status = "completed" if i < len(step_results) else "pending"
            if i < len(step_results) and not step_results[i].get("success", True):
                status = "failed"

            steps.append(StepInfo(
                number=i + 1,
                description=step.get("description", ""),
                status=status,
                code=step.get("code"),
            ))

        return tuple(steps)

    def _extract_artifacts(self, result: dict) -> tuple[ArtifactInfo, ...]:
        """Extract artifact information from result dict."""
        artifacts = []
        raw_artifacts = result.get("artifacts", [])

        for artifact in raw_artifacts:
            artifacts.append(ArtifactInfo(
                id=artifact.get("id", ""),
                name=artifact.get("name", ""),
                artifact_type=artifact.get("type", ""),
                step_number=artifact.get("step_number", 0),
            ))

        return tuple(artifacts)

    def _extract_tables(self, result: dict) -> tuple[str, ...]:
        """Extract created table names from result dict."""
        tables = result.get("tables_created", [])
        if not tables and self._session.datastore:
            table_list = self._session.datastore.list_tables()
            tables = [t['name'] for t in table_list]
        return tuple(tables)
