# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Core mixin: __init__, resources, events, state."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional, Any

import constat.llm
from constat.catalog.api_schema_manager import APISchemaManager
from constat.catalog.preload_cache import MetadataPreloadCache
from constat.catalog.schema_manager import SchemaManager
from constat.core.config import Config
from constat.core.tiered_config import ResolvedConfig
from constat.core.models import Plan
from constat.core.resources import SessionResources
from constat.discovery.concept_detector import ConceptDetector
from constat.discovery.doc_tools import DocumentDiscoveryTools
from constat.embedding_loader import EmbeddingModelLoader
from constat.execution.executor import PythonExecutor
from constat.execution.fact_resolver import FactResolver
from constat.execution.intent_classifier import IntentClassifier
from constat.execution.mode import (
    Phase, TurnIntent, ConversationState, PrimaryIntent,
)
from constat.execution.parallel_scheduler import ExecutionContext
from constat.execution.planner import Planner
from constat.execution.scratchpad import Scratchpad
from constat.providers import TaskRouter
from constat.session._types import (
    SessionConfig, StepEvent, ApprovalCallback, ClarificationCallback,
)
from constat.storage.history import SessionHistory
from constat.storage.learnings import LearningStore
from constat.storage.registry import ConstatRegistry
from constat.storage.registry_datastore import RegistryAwareDataStore

logger = logging.getLogger(__name__)


class CoreMixin:

    def __init__(
        self,
        config: Config,
        session_id: str,
        session_config: Optional[SessionConfig] = None,
        history: Optional[SessionHistory] = None,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        user_id: Optional[str] = None,
        data_dir: Optional[Path] = None,
    ):
        self.session_id = session_id  # Always provided by client

        self.config = config
        self.resolved_config: Optional[ResolvedConfig] = None
        self.session_config = session_config or SessionConfig()
        self.user_id = user_id or "default"
        self.data_dir = data_dir or Path(".constat")

        # Start loading embedding model in background immediately
        # This allows the model to load while other initialization happens
        EmbeddingModelLoader.get_instance().start_loading()

        # Initialize components with timing
        t0 = time.time()
        self.schema_manager = SchemaManager(config)
        self.schema_manager.initialize(progress_callback=progress_callback)
        logger.debug(f"Session init: SchemaManager took {time.time() - t0:.2f}s")

        # API schema manager (GraphQL introspection, REST metadata)
        t0 = time.time()
        self.api_schema_manager = APISchemaManager(config)
        if config.apis:
            self.api_schema_manager.initialize(progress_callback=progress_callback)
        logger.debug(f"Session init: APISchemaManager took {time.time() - t0:.2f}s")

        # Metadata preload cache for faster context loading
        t0 = time.time()
        self.preload_cache = MetadataPreloadCache(config)
        self._preloaded_context: Optional[str] = None
        # noinspection PyUnresolvedReferences
        self._load_preloaded_context()
        logger.debug(f"Session init: MetadataPreloadCache took {time.time() - t0:.2f}s")

        # Document discovery tools (for reference documents)
        t0 = time.time()
        self.doc_tools = DocumentDiscoveryTools(config)
        logger.debug(f"Session init: DocumentDiscoveryTools took {time.time() - t0:.2f}s")

        # Entity extraction is handled by session_manager.refresh_entities_async()
        # after session creation — not during __init__ to avoid dual extraction race
        self._entities_extracted = False

        # Task router for model routing with escalation
        self.router = TaskRouter(config.llm)

        constat.llm.set_backend(self.router)
        # noinspection PyUnresolvedReferences
        constat.llm.on_call(self._handle_llm_call_event)

        # Default provider (for backward compatibility - e.g., fact resolver)
        self.llm = self.router._get_provider(
            self.router.routing_config.get_models_for_task("general")[0]
        )

        self.planner = Planner(
            config, self.schema_manager, self.router,
            doc_tools=self.doc_tools,
            api_schema_manager=self.api_schema_manager,
        )


        self.executor = PythonExecutor(
            timeout_seconds=config.execution.timeout_seconds,
            allowed_imports=config.execution.allowed_imports or None,
        )

        self.history = history or SessionHistory(user_id=self.user_id)

        # Session state
        # Note: self.session_id is set at the top of __init__ (required parameter)
        # Store server_session_id for history mapping (allows find_session_by_server_id)
        self.server_session_id: Optional[str] = session_id  # Server UUID for reverse lookup
        self.plan: Optional[Plan] = None
        self.scratchpad = Scratchpad()
        self.datastore: Optional[RegistryAwareDataStore] = None  # Persistent storage (only shared state between steps)

        # Central registry for tables and artifacts (shared across sessions)
        self.registry = ConstatRegistry(base_dir=self.data_dir)

        # Session-scoped data sources (added via /database and /file commands)
        self.session_databases: dict[str, dict] = {}  # name -> {type, uri, description}
        self.session_files: dict[str, dict] = {}  # name -> {uri, auth, description}

        # Domain APIs (added when domains are activated)
        self._domain_apis: dict[str, Any] = {}  # name -> ApiConfig

        # Consolidated view of all available resources (single source of truth)
        self.resources = SessionResources()
        self._init_resources_from_config()

        # Pass resources to planner (after resources are initialized)
        self.planner.resources = self.resources

        # Fact resolver for auditable mode
        self.fact_resolver = FactResolver(
            llm=self.llm,
            schema_manager=self.schema_manager,
            config=self.config,
            event_callback=self._handle_fact_resolver_event,
            doc_tools=self.doc_tools,  # Enable document-based fact resolution
        )

        # Learning store for corrections and patterns
        self.learning_store = LearningStore(user_id=self.user_id)

        # Pass learning store to planner for injecting learned rules
        self.planner.set_learning_store(self.learning_store)

        # Role manager for user-defined roles ({data_dir}/{user_id}/roles.yaml)
        from constat.core.roles import RoleManager
        self.role_manager = RoleManager(user_id=self.user_id, base_dir=self.data_dir)

        # Role matcher for dynamic role selection based on query
        from constat.core.role_matcher import RoleMatcher
        self.role_matcher = RoleMatcher(self.role_manager)
        # Initialize lazily on first use to avoid blocking startup

        # Skill manager: loads system, domain, and user skills in precedence order
        from constat.core.skills import SkillManager
        system_skills_dir = Path(config.config_dir) / "skills" if config.config_dir else None
        self.skill_manager = SkillManager(
            user_id=self.user_id, base_dir=self.data_dir,
            system_skills_dir=system_skills_dir,
        )

        # Skill matcher for dynamic skill selection based on query
        from constat.core.skill_matcher import SkillMatcher
        self.skill_matcher = SkillMatcher(self.skill_manager)
        # Initialize lazily on first use to avoid blocking startup

        # Wire skill manager into planner so active skills appear in planning prompts
        self.planner.set_skill_manager(self.skill_manager)

        # Track current role for this query (None = shared context)
        self._current_role_id: Optional[str] = None

        # Event callbacks for monitoring
        self._event_handlers: list[Callable[[StepEvent], None]] = []

        # Approval callback (set via set_approval_callback)
        self._approval_callback: Optional[ApprovalCallback] = None

        # Clarification callback (set via set_clarification_callback)
        self._clarification_callback: Optional[ClarificationCallback] = None

        # Tool response cache for schema tools (cleared on refresh)
        self._tool_cache: dict[str, Any] = {}

        # Cached proof result (set after prove_conversation completes)
        self.last_proof_result: Optional[dict] = None

        # Proof user validations and step hints (set during prove flow)
        self._proof_user_validations: list[dict] = []
        self._proof_step_hints: list = []

        # Concept detector for conditional prompt injection
        t0 = time.time()
        self._concept_detector = ConceptDetector()
        self._concept_detector.initialize()
        logger.debug(f"Session init: ConceptDetector took {time.time() - t0:.2f}s")

        # Phase 3: Conversation state and intent classifier
        # Initialize conversation state with idle phase
        self._conversation_state: ConversationState = ConversationState(
            phase=Phase.IDLE,
        )

        # Intent classifier for turn-level intent detection
        # Pass router as LLM provider for fallback classification
        self._intent_classifier: IntentClassifier = IntentClassifier(
            llm_provider=self.router,
        )

        # Phase 4: Execution Control
        # Cancellation flag for stopping execution mid-flight
        self._cancelled: bool = False

        # Intent queue for messages received during execution
        # Queue behavior per intent type:
        # - plan_new: Queue 1, latest wins (new request replaces queued)
        # - control: Queue in order, process after execution
        # - query: No queue, answered in parallel immediately
        # Each entry is a tuple of (TurnIntent, user_input_string)
        self._intent_queue: list[tuple[TurnIntent, str]] = []

        # Execution context for cancellation signaling to scheduler
        self._execution_context: ExecutionContext = ExecutionContext()

    def _init_resources_from_config(self) -> None:
        """Initialize resources from base config."""
        # Add databases from config
        for name, db_config in self.config.databases.items():
            self.resources.add_database(
                name=name,
                description=db_config.description or "",
                db_type=db_config.type or "sql",
                source="config",
            )

        # Add APIs from config
        if self.config.apis:
            for name, api_config in self.config.apis.items():
                self.resources.add_api(
                    name=name,
                    description=api_config.description or "",
                    api_type=api_config.type or "graphql",
                    source="config",
                )

        # Add documents from config
        if self.config.documents:
            for name, doc_config in self.config.documents.items():
                self.resources.add_document(
                    name=name,
                    description=doc_config.description or "",
                    doc_type=doc_config.type or "file",
                    source="config",
                )

    def add_domain_resources(
        self,
        domain_filename: str,
        databases: dict = None,
        apis: dict = None,
        documents: dict = None,
    ) -> None:
        """Add resources from a domain.

        Args:
            domain_filename: Domain filename for source tracking
            databases: Dict of database configs
            apis: Dict of API configs
            documents: Dict of document configs
        """
        source = f"domain:{domain_filename}"

        if databases:
            for name, db_config in databases.items():
                self.resources.add_database(
                    name=name,
                    description=getattr(db_config, 'description', '') or "",
                    db_type=getattr(db_config, 'type', 'sql') or "sql",
                    source=source,
                )

        if apis:
            for name, api_config in apis.items():
                self.resources.add_api(
                    name=name,
                    description=getattr(api_config, 'description', '') or "",
                    api_type=getattr(api_config, 'type', 'graphql') or "graphql",
                    source=source,
                )

        if documents:
            for name, doc_config in documents.items():
                self.resources.add_document(
                    name=name,
                    description=getattr(doc_config, 'description', '') or "",
                    doc_type=getattr(doc_config, 'type', 'file') or "file",
                    source=source,
                )

    def remove_domain_resources(self, domain_filename: str) -> None:
        """Remove all resources from a domain.

        Args:
            domain_filename: Domain filename
        """
        source = f"domain:{domain_filename}"
        self.resources.remove_by_source(source)

    def sync_resources_to_history(self) -> None:
        """Sync current resources to session history (session.json).

        Call this after loading/unloading domains to keep history in sync.
        """
        if self.session_id and self.history:
            self.history.update_resources(
                session_id=self.session_id,
                databases=self.resources.database_names,
                apis=self.resources.api_names,
                documents=self.resources.document_names,
            )

    def set_approval_callback(self, callback: ApprovalCallback) -> None:
        """
        Set the callback for plan approval.

        The callback receives a PlanApprovalRequest and must return a PlanApprovalResponse.

        Args:
            callback: Function that handles approval requests
        """
        self._approval_callback = callback

    def set_clarification_callback(self, callback: ClarificationCallback) -> None:
        """
        Set the callback for requesting clarification on ambiguous questions.

        The callback receives a ClarificationRequest and must return a ClarificationResponse.

        Args:
            callback: Function that handles clarification requests
        """
        self._clarification_callback = callback

    def on_event(self, handler: Callable[[StepEvent], None]) -> None:
        """Register an event handler for step events."""
        self._event_handlers.append(handler)

    def _emit_event(self, event: StepEvent) -> None:
        """Emit an event to all handlers."""
        for handler in self._event_handlers:
            handler(event)

    def _handle_fact_resolver_event(self, event_type: str, data: dict) -> None:
        """Convert fact resolver events to StepEvents and emit them."""
        self._emit_event(StepEvent(
            event_type=event_type,
            step_number=data.get("step", 0),
            data=data,
        ))

    def _sync_user_facts_to_planner(self) -> None:
        """Sync current user facts to the planner for use in planning prompts."""
        try:
            all_facts = self.fact_resolver.get_all_facts()
            # Convert Fact objects to simple name -> value dict
            facts_dict = {name: fact.value for name, fact in all_facts.items()}
            self.planner.set_user_facts(facts_dict)
        except Exception as e:
            logger.debug(f"Failed to sync user facts to planner: {e}")

    def _sync_available_roles_to_planner(self) -> None:
        """Sync available roles to the planner for role-based step assignment."""
        try:
            role_names = self.role_manager.list_roles()  # Returns list[str]
            # Convert to list of dicts with name and description
            roles_list = []
            for name in role_names:
                role = self.role_manager.get_role(name)
                if role:
                    roles_list.append({"name": name, "description": role.description or ""})
            logger.info(f"[ROLES] Syncing {len(roles_list)} roles to planner: {[r['name'] for r in roles_list]}")
            self.planner.set_available_roles(roles_list)
        except Exception as e:
            logger.warning(f"Failed to sync roles to planner: {e}")

    def cancel_execution(self) -> None:
        """
        Cancel the current execution.

        Sets the cancellation flag which will be checked between steps.
        Completed facts and results are preserved; only pending steps are cancelled.

        This method is thread-safe and can be called from another thread
        (e.g., in response to Ctrl+C or user typing "stop").
        """
        self._cancelled = True
        self._execution_context.cancel()

        # Emit cancellation event
        self._emit_event(StepEvent(
            event_type="execution_cancelled",
            step_number=0,
            data={"message": "Execution cancelled by user"}
        ))

        logger.debug("Execution cancellation requested")

    def is_cancelled(self) -> bool:
        """
        Check if cancellation has been requested.

        Returns:
            True if cancellation has been requested.
        """
        return self._cancelled

    def reset_cancellation(self) -> None:
        """
        Reset the cancellation flag.

        Called at the start of a new execution to clear any previous
        cancellation state.
        """
        self._cancelled = False
        self._execution_context.reset()

    def queue_intent(self, intent: TurnIntent, user_input: str) -> bool:
        """
        Queue an intent for processing after current execution completes.

        Queue behavior depends on intent type:
        - plan_new: Queue 1, latest wins (new request replaces queued)
        - control: Queue in order, process after execution
        - query: Not queued - should be handled in parallel immediately

        Args:
            intent: The TurnIntent to queue.
            user_input: The original user input (stored with intent for later processing).

        Returns:
            True if the intent was queued, False if it should be handled immediately.
        """
        if intent.primary == PrimaryIntent.QUERY:
            # Query intents are not queued - they're answered in parallel
            return False

        if intent.primary == PrimaryIntent.PLAN_NEW:
            # Queue 1, latest wins - replace any existing queued plan_new
            self._intent_queue = [
                (i, inp) for i, inp in self._intent_queue
                if i.primary != PrimaryIntent.PLAN_NEW
            ]
            self._intent_queue.append((intent, user_input))
            logger.debug(f"Queued plan_new intent (latest wins), queue size: {len(self._intent_queue)}")
            return True

        if intent.primary == PrimaryIntent.CONTROL:
            # Control intents queue in order
            self._intent_queue.append((intent, user_input))
            logger.debug(f"Queued control intent, queue size: {len(self._intent_queue)}")
            return True

        if intent.primary == PrimaryIntent.PLAN_CONTINUE:
            # Plan continue triggers replan - cancel current and queue
            self.cancel_execution()
            self._intent_queue.append((intent, user_input))
            logger.debug(f"Queued plan_continue intent (triggers replan), queue size: {len(self._intent_queue)}")
            return True

        return False

    def get_queued_intents_count(self) -> int:
        """
        Get the number of queued intents.

        Returns:
            The number of intents waiting to be processed.
        """
        return len(self._intent_queue)

    def process_queued_intents(self) -> list[dict]:
        """
        Process all queued intents after execution completes.

        Intents are processed in queue order. The queue is cleared
        after processing.

        Returns:
            List of result dicts from processing each queued intent.
        """
        if not self._intent_queue:
            return []

        results = []
        queued = list(self._intent_queue)
        self._intent_queue = []

        logger.debug(f"Processing {len(queued)} queued intents")

        for intent, user_input in queued:
            try:
                if intent.primary == PrimaryIntent.PLAN_NEW:
                    # noinspection PyUnresolvedReferences
                    result = self._handle_plan_new_intent(intent, user_input)
                elif intent.primary == PrimaryIntent.PLAN_CONTINUE:
                    # noinspection PyUnresolvedReferences
                    result = self._handle_plan_continue_intent(intent, user_input)
                elif intent.primary == PrimaryIntent.CONTROL:
                    # noinspection PyUnresolvedReferences
                    result = self._handle_control_intent(intent, user_input)
                else:
                    result = {
                        "success": False,
                        "error": f"Unexpected queued intent type: {intent.primary}",
                    }
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing queued intent: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                })

        return results

    def clear_intent_queue(self) -> int:
        """
        Clear all queued intents.

        Returns:
            The number of intents that were cleared.
        """
        count = len(self._intent_queue)
        self._intent_queue = []
        logger.debug(f"Cleared {count} queued intents")
        return count

    def get_execution_context(self) -> ExecutionContext:
        """
        Get the execution context for external monitoring.

        Returns:
            The ExecutionContext that can be used to check/request cancellation.
        """
        return self._execution_context

    def is_executing(self) -> bool:
        """
        Check if execution is currently in progress.

        Returns:
            True if the session is in EXECUTING phase.
        """
        return self._conversation_state.phase == Phase.EXECUTING

    def get_state(self) -> dict:
        """Get current session state for inspection or resumption."""
        return {
            "session_id": self.session_id,
            "plan": self.plan,
            "scratchpad": self.scratchpad.to_markdown(),
            "state": self.datastore.get_all_state() if self.datastore else {},
            "completed_steps": self.plan.completed_steps if self.plan else [],
            "datastore_tables": self.datastore.list_tables() if self.datastore else [],
        }
