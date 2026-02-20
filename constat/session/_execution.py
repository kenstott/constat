# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Execution mixin: step execution, globals, helpers, error learning."""

from __future__ import annotations

import logging
import re
import time

from typing import Callable

from constat.core.models import FailureSuggestion, PostValidation, Step, StepResult, TaskType, ValidationOnFail
from constat.email import create_send_email
from constat.execution import RETRY_PROMPT_TEMPLATE
from constat.execution.executor import format_error_for_retry
from constat.session._types import StepEvent
from constat.storage.learnings import LearningCategory, LearningSource
from constat.visualization import create_viz_helper

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class ExecutionMixin:

    def _create_llm_ask_helper(self) -> Callable:
        """Create a helper function for step code to query LLM for general knowledge."""
        def llm_ask(question: str) -> int | float | str:
            """
            Ask the LLM a general knowledge question.

            Use this for facts not available in the databases, such as:
            - Industry benchmarks and averages
            - General domain knowledge
            - Conversion factors or standard values
            - Definitions and explanations

            Args:
                question: The question to ask

            Returns:
                The value (number, string, or ISO date string)
            """
            return self._resolve_llm_knowledge(question)
        return llm_ask

    def _handle_llm_call_event(self, _event) -> None:
        """Callback from constat.llm primitives — flags LLM knowledge usage."""
        self._inference_used_llm_map = True

    def _create_doc_read_helper(self) -> Callable:
        """Create a helper to read reference documents at execution time."""
        def doc_read(name: str) -> str:
            """Read a reference document by name. Returns the document text content.

            Prints provenance info (source, path, modified date) to stdout so
            consumers can verify the document is current.

            Args:
                name: Document name as configured (e.g., 'compensation_policy', 'business_rules')

            Returns:
                Document text content.

            Raises:
                ValueError: If document not found or has no text content.
            """
            from datetime import datetime

            # Reload fresh to pick up any file changes
            self.doc_tools._load_document_with_mtime(name)

            result = self.doc_tools.get_document(name)
            if "error" in result:
                raise ValueError(f"Document '{name}' not found: {result['error']}")
            content = result.get("content")
            if not content:
                raise ValueError(f"Document '{name}' has no text content (may be binary)")

            # Build provenance info
            provenance_parts = [f"[DOC] Source: {name}"]
            path = result.get("path")
            if path:
                provenance_parts.append(f"Path: {path}")
            doc = self.doc_tools._loaded_documents.get(name)
            if doc and doc.file_mtime:
                mtime_str = datetime.fromtimestamp(doc.file_mtime).strftime("%Y-%m-%d %H:%M:%S")
                provenance_parts.append(f"Modified: {mtime_str}")
            if doc and doc.content_hash:
                provenance_parts.append(f"Hash: {doc.content_hash}")
            print(" | ".join(provenance_parts))

            return content

        return doc_read

    def _is_current_plan_sensitive(self) -> bool:
        """Check if the current plan involves sensitive data."""
        return self.plan is not None and self.plan.contains_sensitive_data

    def _create_publish_helper(self) -> Callable:
        """Create a helper function for step code to publish artifacts.

        Published artifacts appear in the artifacts panel (consequential outputs).
        Unpublished artifacts are still accessible via inline links and /artifacts.
        """
        def publish(name: str, title: str = None, _description: str = None) -> bool:
            """
            Mark an artifact as published for the artifacts panel.

            Call this for artifacts that are consequential outputs (final deliverables)
            rather than intermediate results. Published artifacts appear prominently
            in the artifacts panel.

            Args:
                name: The table or artifact name to publish
                title: Optional human-friendly display title
                _description: Optional description (unused)

            Returns:
                True if published successfully, False if artifact not found

            Example:
                # After creating a final summary table
                store.save_dataframe('executive_summary', summary_df)
                publish('executive_summary', title='Executive Summary Report')
            """
            if not self.registry:
                return False

            # Try to publish as table first
            if self.registry.publish_table(
                self.user_id, self.session_id, name,
                is_published=True, title=title
            ):
                return True

            # Try as artifact
            if self.registry.publish_artifact(
                self.user_id, self.session_id, name,
                is_published=True, title=title
            ):
                return True

            return False

        return publish

    def _get_execution_globals(self) -> dict:
        """Get globals dict for code execution.

        Each step runs in isolation - only `store` (DuckDB) is shared.
        """
        def parse_number(val):
            """Parse string-formatted numbers, handling ranges, series, percentages, currency, and units.

            Returns a tuple of ALL extracted numbers, preserving order.
            Use min()/max() on the result for range bounds.

            Examples:
              "8-12%"           → (8.0, 12.0)
              "5%"              → (5.0,)
              "$1,200"          → (1200.0,)
              "8 to 12"         → (8.0, 12.0)
              "between 5 and 10"→ (5.0, 10.0)
              "1, 2, 3"         → (1.0, 2.0, 3.0)
              "1; 2; 3"         → (1.0, 2.0, 3.0)
              "10k"             → (10000.0,)
              "1.5M"            → (1500000.0,)
              "(5%)"            → (-5.0,)
              "up to 15%"       → (0.0, 15.0)
              None / NaN        → (0.0,)
            """
            import re as _re
            if val is None:
                return (0.0,)
            # Handle numeric passthrough
            if isinstance(val, (int, float)):
                if val != val:  # NaN check
                    return (0.0,)
                return (float(val),)

            s = str(val).strip()
            if not s:
                return (0.0,)

            # Detect accounting-style negatives: (5%) → -5
            is_accounting_neg = s.startswith('(') and s.endswith(')')
            if is_accounting_neg:
                s = s[1:-1].strip()

            # Strip currency symbols, normalize whitespace
            s = _re.sub(r'[£€¥₹]', '', s)
            s = s.replace('$', '').replace('%', '').replace(',', '').strip()

            # Expand unit suffixes: k, M, B, T
            _unit_mult = {'k': 1e3, 'K': 1e3, 'm': 1e6, 'M': 1e6, 'b': 1e9, 'B': 1e9, 't': 1e12, 'T': 1e12}
            def _apply_unit(num_str):
                num_str = num_str.strip()
                if num_str and num_str[-1] in _unit_mult:
                    try:
                        return float(num_str[:-1]) * _unit_mult[num_str[-1]]
                    except ValueError:
                        pass
                try:
                    return float(num_str)
                except ValueError:
                    return None

            # Split on range/list delimiters: -, –, —, to, and, /, ;, ,, |, or
            parts = _re.split(r'\s*[-–—/;,|]\s*|\s+to\s+|\s+and\s+|\s+or\s+', s, flags=_re.IGNORECASE)
            numbers = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                v = _apply_unit(p)
                if v is not None:
                    numbers.append(v)

            # Fallback: regex extract all decimal numbers from original string
            if not numbers:
                for m in _re.finditer(r'-?\d+\.?\d*', s):
                    try:
                        numbers.append(float(m.group()))
                    except ValueError:
                        pass

            if not numbers:
                return (0.0,)

            if is_accounting_neg:
                numbers = [-n for n in numbers]

            # "up to X" → (0, X)
            if _re.match(r'up\s+to', str(val), _re.IGNORECASE) and len(numbers) == 1:
                return 0.0, numbers[0]

            return tuple(numbers)

        import constat.llm

        globals_dict = {
            "store": self.datastore,  # Persistent datastore - only shared state between steps
            "parse_number": parse_number,  # Parse string numbers/ranges: "8-12%" → (8.0, 12.0)
            "llm_ask": self._create_llm_ask_helper(),  # LLM query helper for general knowledge
            "send_email": create_send_email(
                self.config.email,
                is_sensitive=self._is_current_plan_sensitive,
            ),  # Email function - blocked if plan involves sensitive data
            "viz": create_viz_helper(
                datastore=self.datastore,
                print_file_refs=self.config.execution.print_file_refs,
                session_id=self.session_id,
                user_id=self.user_id,
                registry=self.registry,
                open_with_system_viewer=self.config.execution.open_with_system_viewer,
            ),  # Visualization/file output helper
            "publish": self._create_publish_helper(),  # Mark artifact as published for artifacts panel
            "facts": self._get_facts_dict(),  # Resolved facts as dict (loaded from _facts table)
            "llm_map": constat.llm.llm_map,
            "llm_classify": constat.llm.llm_classify,
            "llm_extract": constat.llm.llm_extract,
            "llm_summarize": constat.llm.llm_summarize,
            "llm_score": constat.llm.llm_score,
            "doc_read": self._create_doc_read_helper(),
        }

        # Provide database connections from config
        # SQL databases get both raw engine (db_<name>) and transpiling helper (sql_<name>)
        from constat.catalog.sql_transpiler import TranspilingConnection, create_sql_helper

        config_db_names = set()
        first_db = None
        for i, (db_name, db_config) in enumerate(self.config.databases.items()):
            config_db_names.add(db_name)
            conn = self.schema_manager.get_connection(db_name)

            # For SQL databases wrapped in TranspilingConnection:
            # - db_<name> = raw engine (for pd.read_sql compatibility)
            # - sql_<name> = helper function with auto-transpilation
            if isinstance(conn, TranspilingConnection):
                globals_dict[f"db_{db_name}"] = conn.engine
                globals_dict[f"sql_{db_name}"] = create_sql_helper(conn)
                if i == 0:
                    globals_dict["db"] = conn.engine
                    globals_dict["sql"] = create_sql_helper(conn)
                    first_db = conn.engine
            else:
                globals_dict[f"db_{db_name}"] = conn
                if i == 0:
                    globals_dict["db"] = conn
                    first_db = conn

        # Also include dynamically added databases (from domains) not in config
        for db_name in self.schema_manager.connections.keys():
            if db_name not in config_db_names:
                conn = self.schema_manager.connections[db_name]
                if isinstance(conn, TranspilingConnection):
                    globals_dict[f"db_{db_name}"] = conn.engine
                    globals_dict[f"sql_{db_name}"] = create_sql_helper(conn)
                    if first_db is None:
                        globals_dict["db"] = conn.engine
                        globals_dict["sql"] = create_sql_helper(conn)
                        first_db = conn.engine
                else:
                    globals_dict[f"db_{db_name}"] = conn
                    if first_db is None:
                        globals_dict["db"] = conn
                        first_db = conn
        # noinspection DuplicatedCode
        for db_name in self.schema_manager.nosql_connections.keys():
            if db_name not in config_db_names:
                globals_dict[f"db_{db_name}"] = self.schema_manager.nosql_connections[db_name]
        for db_name in self.schema_manager.file_connections.keys():
            if db_name not in config_db_names:
                conn = self.schema_manager.file_connections[db_name]
                if hasattr(conn, 'path'):
                    globals_dict[f"file_{db_name}"] = conn.path

        # Inject active skill functions into execution namespace
        self._inject_skill_functions(globals_dict)

        # Provide API clients for GraphQL/REST APIs (config + domain APIs)
        all_apis = self.get_all_apis()
        if all_apis:
            from constat.catalog.api_executor import APIExecutor
            # Create executor with merged config (config APIs + domain APIs)
            api_executor = APIExecutor(self.config, domain_apis=self._domain_apis)
            for api_name, api_config in all_apis.items():
                if api_config.type == "graphql":
                    # Create a GraphQL query function
                    globals_dict[f"api_{api_name}"] = lambda query, variables=None, _name=api_name, _exec=api_executor: \
                        _exec.execute_graphql(_name, query, variables)
                else:
                    # Create a REST call function
                    globals_dict[f"api_{api_name}"] = lambda operation, params=None, _name=api_name, _exec=api_executor: \
                        _exec.execute_rest(_name, operation, params or {})

        return globals_dict

    def _inject_skill_functions(self, globals_dict: dict) -> None:
        """Load active skill scripts and inject declared exports into globals_dict.

        Only functions declared in the skill's frontmatter `exports` field are
        injected. Each function is namespaced as `{pkg_name}_{function_name}`.

        Resolves `dependencies` transitively: if skill A depends on B, B's
        exports are loaded first and injected into A's module namespace so
        A's functions can call B's namespaced functions at runtime.
        """
        import importlib.util

        active_skills = self.skill_manager.active_skill_objects
        if not active_skills:
            return

        # 1. Collect all skills needed (active + transitive dependencies)
        all_needed: dict[str, object] = {}  # name -> Skill

        def _collect(skill):
            if skill.name in all_needed:
                return
            all_needed[skill.name] = skill
            for dep_name in skill.dependencies:
                dep = self.skill_manager.get_skill(dep_name)
                if dep:
                    _collect(dep)
                else:
                    logger.warning(f"[SKILL_INJECT] Dependency '{dep_name}' not found for '{skill.name}'")

        for skill in active_skills:
            _collect(skill)

        # 2. Load all skill modules, extract exported functions
        loaded_fns: dict[str, dict[str, object]] = {}   # pkg_name -> {fn_name: callable}
        loaded_modules: dict[str, object] = {}           # pkg_name -> module object

        for skill in all_needed.values():
            if not skill.exports:
                continue

            skill_dir = self.skill_manager.get_skill_dir(skill.name)
            if not skill_dir:
                continue
            scripts_dir = skill_dir / "scripts"
            if not scripts_dir.exists():
                continue

            pkg_name = skill.name.replace("-", "_").replace(" ", "_")
            loaded_fns[pkg_name] = {}

            for export_entry in skill.exports:
                script_name = export_entry.get("script", "")
                fn_names = export_entry.get("functions", [])
                if not script_name or not fn_names:
                    continue

                script_path = scripts_dir / script_name
                if not script_path.exists():
                    logger.warning(f"[SKILL_INJECT] Script not found: {script_path}")
                    continue

                module_name = f"_constat_skill_{pkg_name}_{script_path.stem}"
                try:
                    spec = importlib.util.spec_from_file_location(module_name, script_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    loaded_modules[pkg_name] = module
                except Exception as e:
                    logger.warning(f"[SKILL_INJECT] Failed to load {script_path}: {e}")
                    continue

                for fn_name in fn_names:
                    obj = getattr(module, fn_name, None)
                    if obj is None or not callable(obj):
                        logger.warning(f"[SKILL_INJECT] Function '{fn_name}' not found in {script_name}")
                        continue
                    loaded_fns[pkg_name][fn_name] = obj

        # 3. Inject dependency functions into each skill's module namespace
        #    so that skill A's code can call skill_b_run_proof() as a global
        for skill in all_needed.values():
            if not skill.dependencies:
                continue
            pkg_name = skill.name.replace("-", "_").replace(" ", "_")
            module = loaded_modules.get(pkg_name)
            if not module:
                continue
            for dep_name in skill.dependencies:
                dep_pkg = dep_name.replace("-", "_").replace(" ", "_")
                for fn_name, fn_obj in loaded_fns.get(dep_pkg, {}).items():
                    setattr(module, f"{dep_pkg}_{fn_name}", fn_obj)

        # 4. Extract all exports into globals_dict for step codegen
        for skill in all_needed.values():
            pkg_name = skill.name.replace("-", "_").replace(" ", "_")
            fns = loaded_fns.get(pkg_name, {})
            if not fns:
                continue
            skill_fn_names = []
            for fn_name, fn_obj in fns.items():
                namespaced = f"{pkg_name}_{fn_name}"
                globals_dict[namespaced] = fn_obj
                skill_fn_names.append(namespaced)
            logger.info(f"[SKILL_INJECT] Injected from '{skill.name}': {skill_fn_names}")

    def _get_facts_dict(self) -> dict:
        """Get resolved facts as a simple dict for use in generated code.

        Returns a dict mapping fact names to their values. This dict is injected
        into the execution globals so steps can reference facts['user_email'] etc.
        """
        facts_dict = {}
        try:
            all_facts = self.fact_resolver.get_all_facts()
            for name, fact in all_facts.items():
                if fact and fact.value is not None:
                    facts_dict[name] = fact.value
        except Exception as e:
            logger.debug(f"Error getting facts dict: {e}")
        return facts_dict

    def _materialize_facts_table(self) -> None:
        """Materialize resolved facts as a _facts table in the datastore.

        Creates a table with columns: name, value, source, description
        This table is used by downloaded scripts and for auditing.
        """
        import pandas as pd

        try:
            all_facts = self.fact_resolver.get_all_facts()
            if not all_facts:
                return

            rows = []
            for name, fact in all_facts.items():
                if fact and fact.value is not None:
                    rows.append({
                        "name": name,
                        "value": str(fact.value),
                        "source": fact.source.value if hasattr(fact.source, "value") else str(fact.source),
                        "description": fact.reasoning or "",
                    })

            if rows:
                facts_df = pd.DataFrame(rows)
                self.datastore.save_dataframe(
                    name="_facts",
                    df=facts_df,
                    step_number=0,  # Step 0 = pre-execution
                    description="Resolved facts for this analysis",
                )
                logger.debug(f"[FACTS] Materialized _facts table with {len(rows)} facts")

        except Exception as e:
            logger.warning(f"Failed to materialize _facts table: {e}")

    def _auto_save_results(self, namespace: dict, step_number: int) -> None:
        """
        Auto-save any DataFrames or lists found in the execution namespace.

        This ensures intermediate results are persisted even if the LLM
        forgot to explicitly save them.
        """
        import pandas as pd

        # Skip internal/injected variables
        skip_vars = {"store", "db", "pd", "np", "llm_ask", "send_email", "facts", "viz", "publish", "__builtins__"}
        skip_prefixes = ("db_", "_")

        # Already-saved tables (don't duplicate by name OR by data content)
        existing_tables = self.datastore.list_tables()
        existing_names = {t["name"] for t in existing_tables}

        # Get row counts of existing tables to detect duplicates by size
        # (cheap heuristic - if same rows, likely same data)
        existing_row_counts: dict[int, str] = {}
        for t in existing_tables:
            row_count = t.get("row_count")
            if row_count is not None:
                existing_row_counts[row_count] = t["name"]

        for var_name, value in namespace.items():
            # Skip internal variables
            if var_name in skip_vars or var_name.startswith(skip_prefixes):
                continue

            # Auto-save DataFrames
            if isinstance(value, pd.DataFrame) and var_name not in existing_names:
                # Skip if this data was already saved under a different name
                # (cheap heuristic: same row count = likely duplicate)
                if len(value) in existing_row_counts:
                    logger.debug(f"Skip auto-save of {var_name}: likely duplicate of {existing_row_counts[len(value)]}")
                    continue

                self.datastore.save_dataframe(
                    name=var_name,
                    df=value,
                    step_number=step_number,
                    description=f"Auto-saved from step {step_number}",
                    role_id=self._current_role_id,
                )

            # Auto-save lists (as state, since they might be useful)
            elif isinstance(value, (list, dict)) and len(value) > 0:
                # Check if already saved in state
                existing = self.datastore.get_state(var_name)
                if existing is None:
                    try:
                        # Convert pandas NA values to None for JSON serialization
                        clean_value = self._make_json_serializable(value)
                        self.datastore.set_state(var_name, clean_value, step_number)
                    except Exception as e:
                        logger.debug(f"Skip auto-save of {var_name}: not JSON-serializable: {e}")

    def _execute_step(self, step: Step) -> StepResult:
        """
        Execute a single step with retry on errors.

        Returns:
            StepResult with success/failure info
        """
        from constat.session._types import STEP_SYSTEM_PROMPT

        start_time = time.time()
        last_code = ""
        last_error = None
        pending_learning_context = None  # Track error for potential learning capture

        # Set current role context for this step (facts created will inherit this role_id)
        previous_role_id = self._current_role_id
        self._current_role_id = step.role_id

        self._emit_event(StepEvent(
            event_type="step_start",
            step_number=step.number,
            data={"goal": step.goal}
        ))

        max_attempts = self.session_config.max_retries_per_step
        for attempt in range(1, max_attempts + 1):
            # Emit detailed generating event
            self._emit_event(StepEvent(
                event_type="generating",
                step_number=step.number,
                data={
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "is_retry": attempt > 1,
                    "goal": step.goal[:50] + "..." if len(step.goal) > 50 else step.goal,
                    "retry_reason": last_error[:100] if attempt > 1 and last_error else None,
                }
            ))

            # Use router with step's task_type for automatic model selection/escalation
            if attempt == 1:
                prompt = self._build_step_prompt(step)
                result = self.router.execute_code(
                    task_type=step.task_type,
                    system=STEP_SYSTEM_PROMPT,
                    user_message=prompt,
                    tools=self._get_schema_tools(),
                    tool_handlers=self._get_tool_handlers(),
                    complexity=step.complexity,
                )
            else:
                # Track error context for potential learning capture
                pending_learning_context = {
                    "error_message": last_error[:500] if last_error else "",
                    "original_code": last_code[:500] if last_code else "",
                    "step_goal": step.goal,
                    "attempt": attempt,
                }

                retry_prompt = RETRY_PROMPT_TEMPLATE.format(
                    error_details=last_error,
                    previous_code=last_code,
                )
                result = self.router.execute_code(
                    task_type=step.task_type,
                    system=STEP_SYSTEM_PROMPT,
                    user_message=retry_prompt,
                    tools=self._get_schema_tools(),
                    tool_handlers=self._get_tool_handlers(),
                    complexity=step.complexity,
                )

            if not result.success:
                # Router exhausted all models
                raise RuntimeError(f"Code generation failed: {result.content}")

            code = result.content

            step.code = code

            self._emit_event(StepEvent(
                event_type="executing",
                step_number=step.number,
                data={
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "is_retry": attempt > 1,
                    "code_lines": len(code.split('\n')),
                }
            ))

            # Track tables before execution (name + version to detect updates)
            tables_before_list = self.datastore.list_tables() if self.datastore else []
            tables_before = set(t['name'] for t in tables_before_list)
            versions_before = {t['name']: t.get('version', 1) for t in tables_before_list}

            # Execute
            exec_globals = self._get_execution_globals()
            result = self.executor.execute(code, exec_globals)
            logger.debug(f"[Step {step.number}] Execution result (attempt {attempt}): success={result.success}, error={result.error_message()[:200] if not result.success else 'none'}")

            # Auto-save any DataFrames or lists created during execution
            if result.success and self.datastore:
                self._auto_save_results(result.namespace, step.number)

            # Record artifacts in datastore
            if self.datastore:
                self.datastore.add_artifact(step.number, attempt, "code", code, role_id=self._current_role_id)
                if result.stdout:
                    self.datastore.add_artifact(step.number, attempt, "output", result.stdout, role_id=self._current_role_id)

            if result.success:
                duration_ms = int((time.time() - start_time) * 1000)

                # Capture learning if this was a successful retry
                if attempt > 1 and pending_learning_context:
                    self._capture_error_learning(
                        context=pending_learning_context,
                        fixed_code=code,
                    )

                # Run post-validations
                validation_warnings: list[str] = []
                if step.post_validations:
                    logger.debug(f"[Step {step.number}] Running {len(step.post_validations)} post-validations (attempt {attempt})")
                    validation_warnings, failed_validation = self._run_post_validations(step, result.namespace)
                    logger.debug(f"[Step {step.number}] Post-validation result: failed={failed_validation is not None}, warnings={len(validation_warnings)}")

                    if failed_validation:
                        if failed_validation.on_fail == ValidationOnFail.CLARIFY:
                            clarify_response = self._ask_validation_clarification(
                                step, failed_validation
                            )
                            if clarify_response:
                                last_code = code
                                stdout_hint = f"\nCode stdout:\n{result.stdout[-2000:]}" if result.stdout else ""
                                last_error = f"Validation failed: {failed_validation.description}. User guidance: {clarify_response}{stdout_hint}"
                                continue
                            # User skipped — treat as warning
                            validation_warnings.append(f"Skipped: {failed_validation.description}")

                        elif failed_validation.on_fail == ValidationOnFail.RETRY:
                            last_code = code
                            stdout_context = ""
                            if result.stdout:
                                stdout_context = f"\nCode stdout (shows actual state):\n{result.stdout[-2000:]}\n"
                            last_error = (
                                f"Code executed without errors, but post-validation failed.\n"
                                f"Validation: {failed_validation.description}\n"
                                f"Expression: {failed_validation.expression}\n"
                                f"{stdout_context}"
                                f"The code must be fixed so this validation passes."
                            )
                            self._emit_event(StepEvent(
                                event_type="validation_retry",
                                step_number=step.number,
                                data={"validation": failed_validation.description}
                            ))
                            continue

                    if validation_warnings:
                        self._emit_event(StepEvent(
                            event_type="validation_warnings",
                            step_number=step.number,
                            data={"warnings": validation_warnings}
                        ))

                # Detect new AND updated tables
                tables_after_list = self.datastore.list_tables() if self.datastore else []
                tables_after = set(t['name'] for t in tables_after_list)
                versions_after = {t['name']: t.get('version', 1) for t in tables_after_list}
                new_tables = tables_after - tables_before
                updated_tables = {
                    name for name in tables_before & tables_after
                    if versions_after.get(name, 1) > versions_before.get(name, 1)
                    and not name.startswith('_')  # Skip internal tables
                }
                tables_created = list(new_tables | updated_tables)

                self._emit_event(StepEvent(
                    event_type="step_complete",
                    step_number=step.number,
                    data={
                        "goal": step.goal,
                        "code": code,
                        "stdout": result.stdout,
                        "attempts": attempt,
                        "duration_ms": duration_ms,
                        "tables_created": tables_created,
                    }
                ))

                # Persist step code to disk
                if self.session_id:
                    self.history.save_step_code(
                        session_id=self.session_id,
                        step_number=step.number,
                        goal=step.goal,
                        code=code,
                        output=result.stdout,
                    )

                # Restore previous role context
                self._current_role_id = previous_role_id

                return StepResult(
                    success=True,
                    stdout=result.stdout,
                    attempts=attempt,
                    duration_ms=duration_ms,
                    tables_created=tables_created,
                    code=code,
                    validation_warnings=validation_warnings,
                )

            # Prepare for retry
            last_code = code
            last_error = format_error_for_retry(result, code)

            # Log error for debugging
            logger.warning(f"[Step {step.number}] Execution error (attempt {attempt}/{max_attempts}): {result.error_message() or 'Unknown error'}")
            logger.debug(f"[Step {step.number}] Failed code:\n{code}")

            # Record error artifact
            if self.datastore:
                self.datastore.add_artifact(step.number, attempt, "error", last_error, role_id=self._current_role_id)

            # Determine error type for better status messages
            error_lower = last_error.lower() if last_error else ""
            if "sql" in error_lower or "query" in error_lower:
                error_type = "SQL error"
            elif "name" in error_lower and "not defined" in error_lower:
                error_type = "Variable not found"
            elif "type" in error_lower:
                error_type = "Type error"
            elif "key" in error_lower:
                error_type = "Key error"
            elif "index" in error_lower:
                error_type = "Index error"
            elif "timeout" in error_lower:
                error_type = "Timeout"
            else:
                error_type = "Runtime error"

            will_retry = attempt < max_attempts
            if will_retry:
                logger.debug(f"[Step {step.number}] Will retry ({max_attempts - attempt} attempts remaining)")
            self._emit_event(StepEvent(
                event_type="step_error",
                step_number=step.number,
                data={
                    "error": last_error,
                    "error_type": error_type,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "will_retry": will_retry,
                    "next_attempt": attempt + 1 if will_retry else None,
                }
            ))

        # Max retries exceeded - generate suggestions for alternative approaches
        logger.warning(f"[Step {step.number}] Failed after {max_attempts} attempts: {last_error[:200]}")
        duration_ms = int((time.time() - start_time) * 1000)
        suggestions = self._generate_failure_suggestions(step, last_error, last_code)

        # Emit step_failed event with suggestions
        self._emit_event(StepEvent(
            event_type="step_failed",
            step_number=step.number,
            data={
                "error": last_error,
                "attempts": self.session_config.max_retries_per_step,
                "suggestions": suggestions,
            }
        ))

        # Persist failed step code to disk (with error)
        if self.session_id and last_code:
            self.history.save_step_code(
                session_id=self.session_id,
                step_number=step.number,
                goal=step.goal,
                code=last_code,
                error=last_error,
            )

        # Restore previous role context
        self._current_role_id = previous_role_id

        return StepResult(
            success=False,
            stdout="",
            error=f"Failed after {self.session_config.max_retries_per_step} attempts. Last error: {last_error}",
            attempts=self.session_config.max_retries_per_step,
            duration_ms=duration_ms,
            suggestions=suggestions,
        )

    def _run_post_validations(
        self, step: Step, namespace: dict
    ) -> tuple[list[str], PostValidation | None]:
        """Run post-validations against step's execution namespace.

        Returns:
            (warnings, first_failing_validation)
            - warnings: list of warning messages from on_fail=WARN validations
            - first_failing_validation: first RETRY or CLARIFY validation that failed, or None
        """
        warnings: list[str] = []
        # Inject store tables so validations can reference them by name
        try:
            for t in self.datastore.list_tables():
                name = t["name"]
                if name not in namespace:
                    df = self.datastore.load_dataframe(name)
                    if df is not None:
                        namespace[name] = df
                        logger.debug(f"[Step {step.number}] Injected store table '{name}' into validation namespace")
        except Exception as e:
            logger.warning(f"[Step {step.number}] Failed to inject store tables for validation: {e}")
        for v in step.post_validations:
            try:
                eval_globals = {**namespace, "__builtins__": __builtins__}
                result = eval(v.expression, eval_globals)  # noqa: S307
                passed = bool(result)
            except Exception as e:
                logger.warning(f"[Step {step.number}] Post-validation expression error: {v.expression} -> {e}")
                passed = False

            if not passed:
                if v.on_fail == ValidationOnFail.WARN:
                    warnings.append(f"Validation warning: {v.description}")
                else:
                    # RETRY or CLARIFY — return immediately
                    return warnings, v
        return warnings, None

    def _ask_validation_clarification(self, step: Step, validation: PostValidation) -> str | None:
        """Ask user for clarification when a post-validation fails with on_fail=CLARIFY.

        Returns:
            User's response string, or None if skipped/unavailable.
        """
        from constat.session._types import ClarificationRequest, ClarificationQuestion

        if not self._clarification_callback:
            return None

        question = validation.clarify_question or f"Validation failed: {validation.description}. How should we proceed?"
        request = ClarificationRequest(
            original_question=step.goal,
            ambiguity_reason=f"Post-validation failed: {validation.description}",
            questions=[ClarificationQuestion(text=question)],
        )

        self._emit_event(StepEvent(
            event_type="clarification_needed",
            step_number=step.number,
            data={
                "reason": request.ambiguity_reason,
                "questions": request.questions,
            }
        ))

        response = self._clarification_callback(request)
        if response.skip:
            return None

        # Return first non-empty answer
        for answer in response.answers.values():
            if answer:
                return answer
        return None

    @staticmethod
    def _detect_glossary_correction(correction: str, context: dict) -> dict | None:
        """Detect glossary-relevant corrections from user input or error fixes.

        Returns:
            Detection dict with type "alias" or "definition", or None.
        """
        if not correction:
            return None

        text = correction.strip()

        # --- Alias from user correction: "use X not Y" / "use X instead of Y" ---
        m = re.search(r'use\s+(\w+)\s+(?:not|instead\s+of)\s+(\w+)', text, re.IGNORECASE)
        if m:
            return {"type": "alias", "canonical": m.group(1).lower(), "alternate": m.group(2).lower()}

        # --- Definition from user correction: "X means <long text>" ---
        m = re.search(r'(\w+)\s+means\s+(.{20,})', text, re.IGNORECASE)
        if m:
            return {"type": "definition", "term": m.group(1).lower(), "text": m.group(2).strip()}

        # --- Alias from user correction: "X is Y" / "X means Y" (short RHS = alias) ---
        m = re.search(r'(\w+)\s+(?:is|means)\s+(\w+)$', text, re.IGNORECASE)
        if m:
            return {"type": "alias", "canonical": m.group(2).lower(), "alternate": m.group(1).lower()}

        # --- Alias from error fix: diff SQL identifiers between original and fixed code ---
        original = context.get("original_code", "")
        fixed = context.get("fixed_code", "")
        if original and fixed:
            orig_ids = set(re.findall(r'\b([a-z_][a-z0-9_]*)\b', original.lower()))
            fix_ids = set(re.findall(r'\b([a-z_][a-z0-9_]*)\b', fixed.lower()))
            removed = orig_ids - fix_ids
            added = fix_ids - orig_ids
            if len(removed) == 1 and len(added) == 1:
                wrong = removed.pop()
                correct = added.pop()
                # Skip Python keywords / very short tokens
                if len(wrong) > 2 and len(correct) > 2:
                    return {"type": "alias", "canonical": correct, "alternate": wrong}

        # --- Error-based: "no such table/column" ---
        error_msg = context.get("error_message", "")
        if error_msg and fixed:
            m_err = re.search(r'no such (?:table|column):?\s*["\']?(\w+)', error_msg, re.IGNORECASE)
            if not m_err:
                m_err = re.search(r'(?:table|column)\s+["\']?(\w+)["\']?\s+(?:not found|does not exist)', error_msg, re.IGNORECASE)
            if m_err:
                wrong_name = m_err.group(1).lower()
                fix_ids = set(re.findall(r'\b([a-z_][a-z0-9_]*)\b', fixed.lower()))
                orig_ids = set(re.findall(r'\b([a-z_][a-z0-9_]*)\b', original.lower())) if original else set()
                new_ids = fix_ids - orig_ids
                if len(new_ids) == 1:
                    correct_name = new_ids.pop()
                    if correct_name != wrong_name and len(correct_name) > 2:
                        return {"type": "alias", "canonical": correct_name, "alternate": wrong_name}

        return None

    def _apply_glossary_draft(self, detection: dict) -> None:
        """Write a draft glossary term/alias based on a detected correction."""
        import hashlib
        from constat.discovery.models import GlossaryTerm, display_entity_name

        if not self.doc_tools or not hasattr(self.doc_tools, '_vector_store'):
            return
        vs = self.doc_tools._vector_store
        if vs is None:
            return

        user_id = getattr(self, 'user_id', None)
        session_id = getattr(self, 'session_id', None)
        if not session_id:
            return
        scope_id = user_id or session_id

        def make_id(name: str, domain: str | None = None) -> str:
            key = f"{name}:{scope_id}:{domain or ''}"
            return hashlib.sha256(key.encode()).hexdigest()[:16]

        try:
            dtype = detection["type"]
            if dtype == "alias":
                canonical = detection["canonical"]
                alternate = detection["alternate"]
                existing = vs.get_glossary_term(canonical, session_id, user_id=user_id)
                if existing:
                    aliases = list(existing.aliases or [])
                    if alternate not in aliases:
                        aliases.append(alternate)
                        vs.update_glossary_term(canonical, session_id, {"aliases": aliases}, user_id=user_id)
                else:
                    term = GlossaryTerm(
                        id=make_id(canonical),
                        name=canonical,
                        display_name=display_entity_name(canonical),
                        definition="",
                        aliases=[alternate],
                        status="draft",
                        provenance="learning",
                        session_id=session_id,
                        user_id=user_id or "default",
                    )
                    vs.add_glossary_term(term)
                logger.info(f"[GLOSSARY] Draft alias: {alternate} -> {canonical}")

            elif dtype == "definition":
                term_name = detection["term"]
                definition_text = detection["text"]
                existing = vs.get_glossary_term(term_name, session_id, user_id=user_id)
                if existing:
                    updates = {"definition": definition_text, "provenance": "learning"}
                    vs.update_glossary_term(term_name, session_id, updates, user_id=user_id)
                else:
                    term = GlossaryTerm(
                        id=make_id(term_name),
                        name=term_name,
                        display_name=display_entity_name(term_name),
                        definition=definition_text,
                        status="draft",
                        provenance="learning",
                        session_id=session_id,
                        user_id=user_id or "default",
                    )
                    vs.add_glossary_term(term)
                logger.info(f"[GLOSSARY] Draft definition for: {term_name}")
        except Exception as e:
            logger.debug(f"Glossary draft write failed (non-fatal): {e}")

    def _capture_error_learning(self, context: dict, fixed_code: str) -> None:
        """Capture a learning from a successful error fix.

        Args:
            context: Error context dict with error_message, original_code, step_goal
            fixed_code: The code that successfully fixed the error
        """
        try:
            # Determine category based on error type and context
            category = self._categorize_error(context)

            # Use LLM to generate a concise learning summary
            summary = self._summarize_error_fix(context, fixed_code)
            if not summary:
                # Fallback to a simple summary
                error_preview = context.get("error_message", "")[:100]
                summary = f"Fixed error: {error_preview}"

            # Add fixed code to context
            context["fixed_code"] = fixed_code[:500]

            # Detect glossary-relevant correction
            detection = self._detect_glossary_correction(summary, context)
            if detection:
                category = LearningCategory.GLOSSARY_REFINEMENT
                context["glossary_detection"] = detection

            # Save the learning
            self.learning_store.save_learning(
                category=category,
                context=context,
                correction=summary,
                source=LearningSource.AUTO_CAPTURE,
            )

            # Apply glossary draft after save
            if detection:
                self._apply_glossary_draft(detection)

            # Auto-compact if too many raw learnings
            stats = self.learning_store.get_stats()
            if stats["unpromoted"] >= 50:
                logger.info(f"[LEARNINGS] {stats['unpromoted']} unpromoted learnings — triggering compaction")
                try:
                    self._compact_learnings()
                except Exception as ce:
                    logger.warning(f"[LEARNINGS] Compaction failed (non-fatal): {ce}")
        except Exception as e:
            logger.debug(f"Learning capture failed (non-fatal): {e}")

    def _compact_learnings(self) -> None:
        """Compact raw learnings into rules using LLM to find patterns."""
        unpromoted = [
            l for l in self.learning_store.list_raw_learnings(limit=200, include_promoted=False)
            if not l.get("promoted_to")
        ]
        if len(unpromoted) < 20:
            return

        # Format learnings for LLM
        learning_texts = []
        for l in unpromoted:
            correction = l.get("correction", "")
            category = l.get("category", "")
            ctx = l.get("context", {})
            error = ctx.get("error_message", "")[:100]
            learning_texts.append(f"[{l['id']}] ({category}) {correction} | error: {error}")

        prompt = f"""Group these {len(learning_texts)} code generation learnings into reusable rules.
Each rule should capture a PATTERN that applies across multiple learnings.

Learnings:
{chr(10).join(learning_texts)}

Return JSON array of rules:
[
  {{"summary": "Rule description", "category": "codegen_error", "confidence": 0.85, "source_ids": ["learn_xxx", "learn_yyy"], "tags": ["sql", "duckdb"]}}
]

Guidelines:
- Only create a rule if 3+ learnings share the same pattern
- confidence = fraction of learnings in the group that clearly match
- Learnings not matching any pattern should be omitted (they stay as raw)
- Keep summaries actionable: "Use X instead of Y when Z"

Return ONLY the JSON array."""

        result = self.router.execute(
            task_type=TaskType.SYNTHESIS,
            system="You analyze code error patterns and extract reusable rules.",
            user_message=prompt,
            max_tokens=4096,
        )

        # Parse rules
        import json
        text = result.content.strip()
        if text.startswith("```"):
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
        try:
            rules = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("[LEARNINGS] Failed to parse compaction response as JSON")
            return

        promoted_count = 0
        for rule_data in rules:
            source_ids = rule_data.get("source_ids", [])
            if len(source_ids) < 3:
                continue
            category_str = rule_data.get("category", "codegen_error")
            try:
                category = LearningCategory(category_str)
            except ValueError:
                category = LearningCategory.CODEGEN_ERROR

            rule_id = self.learning_store.save_rule(
                summary=rule_data["summary"],
                category=category,
                confidence=rule_data.get("confidence", 0.8),
                source_learnings=source_ids,
                tags=rule_data.get("tags", []),
            )
            # Archive promoted learnings
            for lid in source_ids:
                self.learning_store.archive_learning(lid, rule_id)
                promoted_count += 1

        logger.info(f"[LEARNINGS] Compacted {promoted_count} learnings into {len(rules)} rules")

    @staticmethod
    def _categorize_error(context: dict) -> LearningCategory:
        """Categorize an error for learning storage.

        Categories:
        - HTTP_ERROR: 4xx/5xx errors from external API calls
        - EXTERNAL_API_ERROR: Other errors in API integration code
        - CODEGEN_ERROR: General code generation errors (default)
        """
        error_msg = context.get("error_message", "").lower()
        original_code = context.get("original_code", "").lower()
        step_goal = context.get("step_goal", "").lower()

        # Check for HTTP errors (4xx/5xx)
        http_error_patterns = [
            "status code 4", "status code 5",
            "status_code=4", "status_code=5",
            "http 4", "http 5",
            "400 ", "401 ", "403 ", "404 ", "405 ",
            "500 ", "502 ", "503 ", "504 ",
            "bad request", "unauthorized", "forbidden", "not found",
            "internal server error", "bad gateway", "service unavailable",
            "httperror", "httpstatuserror",
        ]
        for pattern in http_error_patterns:
            if pattern in error_msg:
                return LearningCategory.HTTP_ERROR

        # Check for external API integration errors (not HTTP status errors)
        api_indicators = [
            "requests.", "httpx.", "aiohttp.",
            "rest api", "graphql", "openapi",
            "api_client", "api_response",
            "response.json()", "response.text",
        ]
        is_api_code = any(ind in original_code for ind in api_indicators)

        # Also check step goal for API-related work
        api_goal_keywords = ["fetch", "call api", "api request", "rest", "graphql", "webhook"]
        is_api_goal = any(kw in step_goal for kw in api_goal_keywords)

        if is_api_code or is_api_goal:
            return LearningCategory.EXTERNAL_API_ERROR

        # Default to general code generation error
        return LearningCategory.CODEGEN_ERROR

    def _save_correction_as_learning(self, user_input: str) -> None:
        """Save a user correction as a reusable learning.

        Captures context from the current session to make the correction
        applicable to future similar queries.

        Args:
            user_input: The user's correction (e.g., "always use customer_id not cust_id")
        """
        if not self.learning_store:
            logger.debug("No learning store available, skipping correction capture")
            return

        try:
            # Build context from current session
            context = {}

            # Add current problem/query context
            if self.datastore:
                problem = self.datastore.get_session_meta("problem")
                if problem:
                    context["original_problem"] = problem[:500]

                # Add available tables for schema context
                tables = self.datastore.list_tables()
                if tables:
                    context["tables"] = [t.get("name", "") for t in tables[:10]]

                # Add any active schemas from catalog
                if hasattr(self, 'catalog') and self.catalog:
                    schemas = getattr(self.catalog, 'get_active_schemas', lambda: [])()
                    if schemas:
                        context["schemas"] = schemas[:5]

            # Add session ID for traceability
            if self.session_id:
                context["session_id"] = self.session_id

            # Detect glossary-relevant correction
            detection = self._detect_glossary_correction(user_input, context)
            category = LearningCategory.USER_CORRECTION
            if detection:
                category = LearningCategory.GLOSSARY_REFINEMENT
                context["glossary_detection"] = detection

            # Save the correction as a learning
            learning_id = self.learning_store.save_learning(
                category=category,
                context=context,
                correction=user_input,
                source=LearningSource.NL_DETECTION,
            )

            # Apply glossary draft after save
            if detection:
                self._apply_glossary_draft(detection)

            logger.info(f"Saved user correction as learning {learning_id}: {user_input[:50]}...")

            # Emit event so UI can acknowledge
            self._emit_event(StepEvent(
                event_type="correction_saved",
                step_number=0,
                data={"correction": user_input, "learning_id": learning_id}
            ))

        except Exception as e:
            logger.debug(f"Correction capture failed (non-fatal): {e}")

    @staticmethod
    def _generate_failure_suggestions(
        step: "Step", error: str, _code: str
    ) -> list["FailureSuggestion"]:
        """Generate suggestions for alternative approaches when a step fails.

        Differentiates between:
        - Codegen failures: LLM couldn't produce working code (need different approach)
        - Runtime errors: Code ran but failed on data/environment (user can redirect)

        Args:
            step: The step that failed
            error: The last error message
            code: The last code that was attempted

        Returns:
            List of FailureSuggestion objects
        """
        from constat.core.models import FailureSuggestion

        suggestions = []
        error_lower = error.lower() if error else ""
        goal_lower = step.goal.lower() if step.goal else ""

        # Detect if this is a codegen failure vs runtime error
        codegen_indicators = [
            "syntax error", "invalid syntax", "unexpected token",
            "code generation failed", "could not generate",
            "parsing error", "indentation error"
        ]
        is_codegen_failure = any(ind in error_lower for ind in codegen_indicators)

        if is_codegen_failure:
            # Codegen failures - LLM couldn't produce working code
            suggestions.extend([
                FailureSuggestion(
                    id="break_down",
                    label="Break into smaller steps",
                    description="Split this step into simpler sub-steps that may be easier to generate",
                    action="break_down"
                ),
                FailureSuggestion(
                    id="simplify_goal",
                    label="Simplify the goal",
                    description="Rephrase the step goal in simpler terms",
                    action="rephrase"
                ),
                FailureSuggestion(
                    id="provide_code",
                    label="Provide code snippet",
                    description="Give a working code example or pattern to follow",
                    action="provide_code"
                ),
                FailureSuggestion(
                    id="report_issue",
                    label="Report this issue",
                    description="This appears to be a code generation bug - report it for investigation",
                    action="report"
                ),
            ])
        else:
            # Runtime errors - likely data/source issues user can redirect

            # Document/search related failures
            if any(term in error_lower or term in goal_lower for term in [
                "document", "search", "not found", "no results", "relevance", "policy", "guideline"
            ]):
                suggestions.extend([
                    FailureSuggestion(
                        id="rephrase_search",
                        label="Rephrase search query",
                        description="Try searching with different or broader terms",
                        action="rephrase"
                    ),
                    FailureSuggestion(
                        id="list_documents",
                        label="List available documents",
                        description="Show all documents so you can specify which one to use",
                        action="list_docs"
                    ),
                    FailureSuggestion(
                        id="load_full_doc",
                        label="Load full document",
                        description="Load an entire document instead of searching chunks",
                        action="load_doc"
                    ),
                ])

            # Database/query related failures
            if any(term in error_lower for term in [
                "table", "column", "sql", "query", "database", "no such", "does not exist"
            ]):
                suggestions.extend([
                    FailureSuggestion(
                        id="list_tables",
                        label="List available tables",
                        description="Show database schema to find correct table/column names",
                        action="list_tables"
                    ),
                    FailureSuggestion(
                        id="different_table",
                        label="Use different data source",
                        description="Specify which table or data source to use instead",
                        action="redirect"
                    ),
                ])

            # Data not found / empty results
            if any(term in error_lower for term in [
                "empty", "no data", "zero rows", "none", "missing"
            ]):
                suggestions.extend([
                    FailureSuggestion(
                        id="broaden_query",
                        label="Broaden the query",
                        description="Remove filters or expand date range to find data",
                        action="broaden"
                    ),
                    FailureSuggestion(
                        id="check_filters",
                        label="Check filter criteria",
                        description="The filters may be too restrictive",
                        action="check_filters"
                    ),
                ])

            # API/connection errors
            if any(term in error_lower for term in [
                "api", "connection", "timeout", "rate limit", "unauthorized", "403", "401", "500"
            ]):
                suggestions.extend([
                    FailureSuggestion(
                        id="use_cached",
                        label="Use cached/local data",
                        description="Check if we have data from a previous call or local source",
                        action="use_cache"
                    ),
                ])

        # Import/syntax errors (could be either codegen or config issue)
        if any(term in error_lower for term in [
            "import", "module", "not allowed"
        ]):
            suggestions.append(
                FailureSuggestion(
                    id="simplify_code",
                    label="Simplify approach",
                    description="Use a simpler method that doesn't require this import",
                    action="simplify"
                )
            )

        # Always offer these general options
        suggestions.extend([
            FailureSuggestion(
                id="modify_step",
                label="Modify this step",
                description="Change the goal or approach for this step",
                action="modify"
            ),
            FailureSuggestion(
                id="skip_step",
                label="Skip this step",
                description="Continue without this step (may affect later steps)",
                action="skip"
            ),
            FailureSuggestion(
                id="manual_input",
                label="Provide value manually",
                description="Enter the needed information directly",
                action="manual"
            ),
        ])

        return suggestions

    def _summarize_error_fix(self, context: dict, fixed_code: str) -> str:
        """Use LLM to generate a concise learning summary from an error fix.

        Args:
            context: Error context with error_message, original_code
            fixed_code: The code that fixed the error

        Returns:
            A concise summary of what was learned, or empty string on failure
        """
        try:
            prompt = f"""Summarize what was learned from this error fix in ONE sentence.

Error: {context.get('error_message', '')[:300]}
Original code snippet: {context.get('original_code', '')[:200]}
Fixed code snippet: {fixed_code[:200]}

Output ONLY a single sentence describing the lesson learned, e.g., "Always use X instead of Y when..."
Do not include any explanation or extra text."""

            response = self.llm.generate(
                system="You are a technical writer summarizing coding lessons learned.",
                user_message=prompt,
                max_tokens=self.router.max_output_tokens,
            )
            # generate() returns string directly
            return response.strip()
        except Exception as e:
            logger.debug(f"Failed to summarize learning (non-fatal): {e}")
            return ""
