# Copyright (c) 2025 Kenneth Stott
# Canary: feb2fd72-3f7f-46ee-9264-38fdfc49de4b
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Session class — GraphQL+SSE execution, queries, and mutations."""

from __future__ import annotations

import asyncio
import io
import json
from typing import Any

import polars

from .entity_cache import _cache as _entity_cache, inflate_glossary
from .graphql import (
    GraphQLClient,
    # Subscription
    QUERY_EXECUTION_SUBSCRIPTION,
    # Execution mutations
    SUBMIT_QUERY, CANCEL_EXECUTION, APPROVE_PLAN,
    ANSWER_CLARIFICATION, SKIP_CLARIFICATION,
    REPLAN_FROM, EDIT_OBJECTIVE, DELETE_OBJECTIVE,
    # Read queries
    SESSION_QUERY, TABLES_QUERY, ARTIFACTS_QUERY, ARTIFACT_QUERY,
    FACTS_QUERY, ENTITIES_QUERY, STEPS_QUERY, INFERENCE_CODES_QUERY,
    SCRATCHPAD_QUERY, SESSION_DDL_QUERY, EXECUTION_OUTPUT_QUERY,
    PROOF_TREE_QUERY, MESSAGES_QUERY, PROMPT_CONTEXT_QUERY,
    DATA_SOURCES_QUERY, DATABASES_QUERY, GLOSSARY_QUERY, GLOSSARY_TERM_QUERY,
    EXECUTION_PLAN_QUERY, OBJECTIVES_QUERY, AGENTS_QUERY,
    # Write mutations
    ADD_FACT, FORGET_FACT,
    CREATE_GLOSSARY_TERM, DELETE_GLOSSARY_TERM, REFINE_DEFINITION, GENERATE_GLOSSARY,
    TOGGLE_TABLE_STAR, DELETE_TABLE, TOGGLE_ARTIFACT_STAR, DELETE_ARTIFACT,
    ADD_DATABASE, REMOVE_DATABASE, ADD_API, REMOVE_API, ADD_DOCUMENT_URI,
    SET_ACTIVE_DOMAINS, RESET_CONTEXT, FLAG_ANSWER,
    CREATE_AGENT, UPDATE_AGENT, DELETE_AGENT, DRAFT_AGENT,
    HANDBOOK_QUERY,
)
from .models import Artifact, ConstatError, SolveResult, StepInfo
from .progress import PrintProgress

try:
    from .widgets import HAS_WIDGETS, widget_plan_approval, widget_clarification, WidgetProgress
except ImportError:
    HAS_WIDGETS = False


def _strip_trailing_json(text: str) -> str:
    """Remove JSON arrays/objects from server output."""
    import re
    lines = text.split('\n')
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if (stripped.startswith('[{') or stripped.startswith('{"')) and (
            stripped.endswith('}]') or stripped.endswith('}')
        ):
            try:
                json.loads(stripped)
                continue
            except json.JSONDecodeError:
                pass
        kept.append(line)
    result = '\n'.join(kept)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def _extract_plan(data: dict) -> tuple[str, list[dict]]:
    """Extract problem and steps from plan_ready event data."""
    plan = data.get("plan", {})
    if plan and plan.get("steps"):
        return plan.get("problem", ""), plan["steps"]
    return data.get("problem", ""), data.get("steps", [])


def _prompt_plan_approval(data: dict) -> dict:
    """Prompt user to approve/reject the plan via input()."""
    _, steps = _extract_plan(data)
    if not steps:
        print("\n[plan] Empty plan — no steps generated. Rejecting.")
        return {"approved": False, "feedback": "Empty plan"}
    answer = input("\nApprove? [Y/n/feedback]: ").strip()
    if not answer or answer.lower() in ("y", "yes"):
        return {"approved": True}
    elif answer.lower() in ("n", "no"):
        return {"approved": False, "feedback": "Rejected by user"}
    else:
        return {"approved": False, "feedback": answer}


def _prompt_clarifications(data: dict) -> dict[str, str] | None:
    """Prompt user for clarification answers via input()."""
    questions = data.get("questions", [])
    if not questions:
        return None
    print(f"\n[clarification] {data.get('ambiguity_reason', 'Clarification needed')}:")
    answers = {}
    for q in questions:
        text = q.get("text", "")
        suggestions = q.get("suggestions", [])
        if suggestions:
            print(f"  {text}")
            for i, s in enumerate(suggestions, 1):
                print(f"    {i}. {s}")
            answer = input("  > ")
            try:
                idx = int(answer) - 1
                if 0 <= idx < len(suggestions):
                    answer = suggestions[idx]
            except ValueError:
                pass
        else:
            answer = input(f"  {text} > ")
        answers[text] = answer
    return answers


class Session:
    """A Constat session — all operations via GraphQL+SSE."""

    _css_injected = False

    def __init__(self, gql: GraphQLClient, http: Any, session_id: str):
        self._gql = gql
        self._http = http  # httpx.Client for binary downloads only
        self.session_id = session_id
        self._progress = None
        self._inject_css()

    def _new_progress(self):
        self._progress = WidgetProgress() if HAS_WIDGETS else PrintProgress()

    @classmethod
    def _inject_css(cls) -> None:
        if cls._css_injected:
            return
        cls._css_injected = True
        try:
            from IPython.display import display, HTML
            display(HTML(
                "<style>"
                ".dt-buttons .dt-button{font-size:11px;padding:2px 8px;}"
                ".dt-container{display:inline-block;min-width:0;}"
                "</style>"
            ))
        except ImportError:
            pass

    # ── Query execution ──────────────────────────────────────────────────

    async def solve(self, question: str, auto_approve: bool = True, require_approval: bool | None = None, timeout: float = 600) -> SolveResult:
        return await self._execute_async(question, is_followup=False, auto_approve=auto_approve, require_approval=require_approval, timeout=timeout)

    async def follow_up(self, question: str, auto_approve: bool = True, require_approval: bool | None = None, timeout: float = 600) -> SolveResult:
        return await self._execute_async(question, is_followup=True, auto_approve=auto_approve, require_approval=require_approval, timeout=timeout)

    async def replay(self, question: str, auto_approve: bool = True, timeout: float = 600, objective_index: int | None = None) -> SolveResult:
        return await self._execute_async(question, is_followup=False, auto_approve=auto_approve, timeout=timeout, replay=True, objective_index=objective_index)

    async def reason_chain(self, question: str, auto_approve: bool = True, require_approval: bool | None = None, timeout: float = 600) -> SolveResult:
        return await self._execute_async(question, is_followup=False, auto_approve=auto_approve, require_approval=require_approval, timeout=timeout)

    async def _execute_async(
        self,
        question: str,
        is_followup: bool,
        auto_approve: bool,
        timeout: float,
        require_approval: bool | None = None,
        replay: bool = False,
        objective_index: int | None = None,
    ) -> SolveResult:
        self._new_progress()
        pre_tables = self._table_names()
        pre_artifact_ids = self._artifact_ids()

        # Build submitQuery input
        input_vars: dict[str, Any] = {"problem": question, "isFollowup": is_followup}
        if require_approval is not None:
            input_vars["requireApproval"] = require_approval
        if replay:
            input_vars["replay"] = True
        if objective_index is not None:
            input_vars["objectiveIndex"] = objective_index

        result = await self._subscribe_and_execute(
            mutation=SUBMIT_QUERY,
            mutation_vars={"sessionId": self.session_id, "input": input_vars},
            auto_approve=auto_approve,
            timeout=timeout,
        )

        if result.success:
            result.tables = self._fetch_new_tables(pre_tables)
            result.artifacts = self._fetch_new_artifacts(pre_artifact_ids)
            result._session = self
        return result

    async def _subscribe_and_execute(
        self,
        mutation: str,
        mutation_vars: dict,
        auto_approve: bool,
        timeout: float,
    ) -> SolveResult:
        """Start SSE subscription, fire mutation, process events until terminal."""
        result = SolveResult(success=False, answer="")

        # We need to start the subscription BEFORE the mutation fires.
        # Use an asyncio.Task for the subscription, signal when ready.
        subscription_started = asyncio.Event()
        events: asyncio.Queue[dict] = asyncio.Queue()

        async def _drain_sse():
            first = True
            async for event_data in self._gql.subscribe_sse(
                QUERY_EXECUTION_SUBSCRIPTION,
                {"sessionId": self.session_id},
                timeout=timeout,
            ):
                if first:
                    subscription_started.set()
                    first = True
                # Extract the subscription field (queryExecution)
                sub_event = event_data.get("query_execution", event_data)
                await events.put(sub_event)
                # Check for terminal events
                et = sub_event.get("event_type", "")
                if et in ("query_complete", "query_error", "query_cancelled"):
                    return
            # SSE stream ended without terminal event
            subscription_started.set()

        sse_task = asyncio.create_task(_drain_sse())

        try:
            # Wait for first SSE event (welcome) before firing mutation
            await asyncio.wait_for(subscription_started.wait(), timeout=15)

            # Fire the mutation
            resp = await self._gql.query_async(mutation, mutation_vars)
            submit_result = resp.get("submit_query", resp)
            # Slash commands may complete synchronously
            if submit_result.get("status") == "completed":
                result.success = True
                result.answer = submit_result.get("message", "")
                sse_task.cancel()
                return result
            elif submit_result.get("status") == "error":
                result.error = submit_result.get("message", "Unknown error")
                sse_task.cancel()
                return result

            # Process events until terminal
            while True:
                try:
                    event = await asyncio.wait_for(events.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    raise ConstatError(f"Timed out after {timeout}s")

                et = event.get("event_type", "")
                data = event.get("data", {})
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        data = {}

                # step_number lives on the event envelope, not inside data
                if "step_number" in event and isinstance(data, dict):
                    data.setdefault("step_number", event["step_number"])

                await self._handle_event(et, data, auto_approve, result)

                if et in ("query_complete", "query_error", "query_cancelled"):
                    break

        finally:
            sse_task.cancel()
            try:
                await sse_task
            except asyncio.CancelledError:
                pass

        return result

    async def _handle_event(self, event_type: str, data: dict, auto_approve: bool, result: SolveResult) -> None:
        """Process a single execution event."""
        server_auto_approved = (event_type == "plan_ready" and data.get("auto_approved"))
        needs_client_approval = not auto_approve and not server_auto_approved

        if HAS_WIDGETS and needs_client_approval and event_type in ("planning_start", "clarification_needed"):
            pass
        else:
            self._progress.handle_event(event_type, data)

        if event_type == "plan_ready":
            if auto_approve or server_auto_approved:
                if not server_auto_approved:
                    self._gql.query(APPROVE_PLAN, {
                        "sessionId": self.session_id,
                        "input": {"approved": True},
                    })
            else:
                if HAS_WIDGETS:
                    decision = await widget_plan_approval(data)
                else:
                    loop = asyncio.get_event_loop()
                    decision = await loop.run_in_executor(None, _prompt_plan_approval, data)
                if decision.get("approved"):
                    input_data: dict[str, Any] = {"approved": True}
                    if decision.get("deleted_steps"):
                        input_data["deletedSteps"] = decision["deleted_steps"]
                    if decision.get("edited_steps"):
                        input_data["editedSteps"] = [
                            {"number": s["number"], "goal": s["goal"]}
                            for s in decision["edited_steps"]
                        ]
                    self._gql.query(APPROVE_PLAN, {"sessionId": self.session_id, "input": input_data})
                else:
                    self._gql.query(APPROVE_PLAN, {
                        "sessionId": self.session_id,
                        "input": {"approved": False, "feedback": decision.get("feedback", "")},
                    })

        elif event_type == "clarification_needed":
            if HAS_WIDGETS:
                answers = await widget_clarification(data)
            else:
                loop = asyncio.get_event_loop()
                answers = await loop.run_in_executor(None, _prompt_clarifications, data)
            if answers:
                self._gql.query(ANSWER_CLARIFICATION, {
                    "sessionId": self.session_id,
                    "answers": answers,
                    "structuredAnswers": answers if isinstance(answers, dict) else {},
                })
            else:
                self._gql.query(SKIP_CLARIFICATION, {"sessionId": self.session_id})

        elif event_type == "entity_state":
            _entity_cache.set(self.session_id, data.get("state", {}), data.get("version", 0))

        elif event_type == "entity_patch":
            _entity_cache.apply_patch(self.session_id, data.get("patch", []), data.get("version", 0))

        elif event_type == "query_complete":
            result.success = True
            raw_output = data.get("output", "")
            result.raw_output = raw_output
            result.answer = _strip_trailing_json(raw_output)
            result.suggestions = data.get("suggestions", [])

        elif event_type == "query_error":
            result.error = data.get("error", "Unknown error")

        elif event_type == "query_cancelled":
            result.error = "Query was cancelled"

        elif event_type == "step_complete":
            result.steps.append(StepInfo(
                number=data.get("step_number", 0),
                goal=data.get("goal", ""),
                status="complete",
                duration_ms=data.get("duration_ms"),
            ))

        elif event_type in ("step_error", "step_failed"):
            result.steps.append(StepInfo(
                number=data.get("step_number", 0),
                goal=data.get("goal", ""),
                status="error",
                error=data.get("error"),
            ))

    # ── Commands ─────────────────────────────────────────────────────────

    async def command(self, cmd: str, timeout: float = 120) -> SolveResult:
        return await self._execute_async(cmd, is_followup=False, auto_approve=True, timeout=timeout)

    async def compact(self) -> SolveResult:
        return await self.command("/compact")

    async def save_plan(self, name: str) -> SolveResult:
        return await self.command(f"/save {name}")

    async def share_plan(self, name: str) -> SolveResult:
        return await self.command(f"/share {name}")

    async def list_plans(self) -> SolveResult:
        return await self.command("/plans")

    async def replay_plan(self, name: str, auto_approve: bool = True, timeout: float = 600) -> SolveResult:
        return await self._execute_async(f"/replay {name}", is_followup=False, auto_approve=auto_approve, timeout=timeout)

    async def summarize(self, target: str) -> SolveResult:
        return await self.command(f"/summarize {target}")

    async def discover(self, query: str) -> SolveResult:
        return await self.command(f"/discover {query}")

    async def redo(self, instruction: str | None = None, auto_approve: bool = True, timeout: float = 600) -> SolveResult:
        cmd = f"/redo {instruction}" if instruction else "/redo"
        return await self._execute_async(cmd, is_followup=True, auto_approve=auto_approve, timeout=timeout)

    async def audit(self) -> SolveResult:
        return await self.command("/audit")

    async def set_verbose(self, on: bool | None = None) -> SolveResult:
        arg = " on" if on is True else " off" if on is False else ""
        return await self.command(f"/verbose{arg}")

    async def set_raw(self, on: bool | None = None) -> SolveResult:
        arg = " on" if on is True else " off" if on is False else ""
        return await self.command(f"/raw{arg}")

    async def set_insights(self, on: bool | None = None) -> SolveResult:
        arg = " on" if on is True else " off" if on is False else ""
        return await self.command(f"/insights{arg}")

    async def preferences(self) -> SolveResult:
        return await self.command("/preferences")

    async def objectives(self) -> SolveResult:
        return await self.command("/objectives")

    # ── Step & objective manipulation (via mutations + SSE) ──────────────

    async def step_edit(self, step_number: int, goal: str, auto_approve: bool = True) -> SolveResult:
        return await self._mutation_with_sse(
            REPLAN_FROM,
            {"sessionId": self.session_id, "stepNumber": step_number, "mode": "edit", "editedGoal": goal},
            auto_approve=auto_approve,
        )

    async def step_delete(self, step_number: int, auto_approve: bool = True) -> SolveResult:
        return await self._mutation_with_sse(
            REPLAN_FROM,
            {"sessionId": self.session_id, "stepNumber": step_number, "mode": "delete"},
            auto_approve=auto_approve,
        )

    async def step_redo(self, step_number: int, auto_approve: bool = True) -> SolveResult:
        return await self._mutation_with_sse(
            REPLAN_FROM,
            {"sessionId": self.session_id, "stepNumber": step_number, "mode": "redo"},
            auto_approve=auto_approve,
        )

    async def edit_objective(self, index: int, text: str, auto_approve: bool = True) -> SolveResult:
        return await self._mutation_with_sse(
            EDIT_OBJECTIVE,
            {"sessionId": self.session_id, "objectiveIndex": index, "newText": text},
            auto_approve=auto_approve,
        )

    async def delete_objective(self, index: int, auto_approve: bool = True) -> SolveResult:
        return await self._mutation_with_sse(
            DELETE_OBJECTIVE,
            {"sessionId": self.session_id, "objectiveIndex": index},
            auto_approve=auto_approve,
        )

    async def _mutation_with_sse(self, mutation: str, variables: dict, auto_approve: bool = True, timeout: float = 300) -> SolveResult:
        """Fire a mutation that triggers re-execution, subscribe for events."""
        self._new_progress()
        pre_tables = self._table_names()
        pre_artifact_ids = self._artifact_ids()
        result = await self._subscribe_and_execute(
            mutation=mutation,
            mutation_vars=variables,
            auto_approve=auto_approve,
            timeout=timeout,
        )
        if result.success:
            result.tables = self._fetch_new_tables(pre_tables)
            result.artifacts = self._fetch_new_artifacts(pre_artifact_ids)
            result._session = self
        return result

    # ── Read queries ─────────────────────────────────────────────────────

    @property
    def status(self) -> dict:
        data = self._gql.query(SESSION_QUERY, {"sessionId": self.session_id})
        return data.get("session", {})

    def tables(self) -> list[dict]:
        data = self._gql.query(TABLES_QUERY, {"sessionId": self.session_id})
        return data.get("tables", {}).get("tables", [])

    def artifacts(self) -> list[Artifact]:
        data = self._gql.query(ARTIFACTS_QUERY, {"sessionId": self.session_id})
        return [
            Artifact(
                id=a["id"], name=a["name"], artifact_type=a["artifact_type"],
                mime_type=a.get("mime_type"), is_starred=a.get("is_starred", False),
            )
            for a in data.get("artifacts", {}).get("artifacts", [])
        ]

    def artifact(self, artifact_id: int) -> Artifact:
        data = self._gql.query(ARTIFACT_QUERY, {"sessionId": self.session_id, "artifactId": artifact_id})
        a = data.get("artifact", {})
        return Artifact(
            id=a["id"], name=a["name"], artifact_type=a["artifact_type"],
            content=a.get("content"), mime_type=a.get("mime_type"),
        )

    def facts(self) -> list[dict]:
        data = self._gql.query(FACTS_QUERY, {"sessionId": self.session_id})
        return data.get("facts", {}).get("facts", [])

    def entities(self) -> list[dict]:
        data = self._gql.query(ENTITIES_QUERY, {"sessionId": self.session_id})
        return data.get("entities", {}).get("entities", [])

    def steps(self) -> list[dict]:
        data = self._gql.query(STEPS_QUERY, {"sessionId": self.session_id})
        return data.get("steps", {}).get("steps", [])

    def inference_codes(self) -> list[dict]:
        data = self._gql.query(INFERENCE_CODES_QUERY, {"sessionId": self.session_id})
        return data.get("inference_codes", {}).get("codes", [])

    def plan(self) -> dict:
        data = self._gql.query(EXECUTION_PLAN_QUERY, {"sessionId": self.session_id})
        return data.get("execution_plan", {})

    def scratchpad(self) -> dict:
        data = self._gql.query(SCRATCHPAD_QUERY, {"sessionId": self.session_id})
        return data.get("scratchpad", {})

    def has_scratchpad(self) -> bool:
        try:
            data = self.scratchpad()
            entries = data.get("entries", data.get("scratchpad", []))
            return bool(entries)
        except Exception:
            return False

    def ddl(self) -> str:
        data = self._gql.query(SESSION_DDL_QUERY, {"sessionId": self.session_id})
        return data.get("session_ddl", "")

    def output(self) -> dict:
        data = self._gql.query(EXECUTION_OUTPUT_QUERY, {"sessionId": self.session_id})
        return data.get("execution_output", {})

    def proof_tree(self) -> dict:
        data = self._gql.query(PROOF_TREE_QUERY, {"sessionId": self.session_id})
        return data.get("proof_tree", {})

    def messages(self) -> list[dict]:
        data = self._gql.query(MESSAGES_QUERY, {"sessionId": self.session_id})
        return data.get("messages", {}).get("messages", [])

    def context(self) -> dict:
        data = self._gql.query(PROMPT_CONTEXT_QUERY, {"sessionId": self.session_id})
        return data.get("prompt_context", {})

    def sources(self) -> dict:
        data = self._gql.query(DATA_SOURCES_QUERY, {"sessionId": self.session_id})
        return data.get("data_sources", {})

    def databases(self) -> list[dict]:
        data = self._gql.query(DATABASES_QUERY, {"sessionId": self.session_id})
        return data.get("databases", {}).get("databases", [])

    def glossary(self, cached: bool = False) -> list[dict]:
        if cached:
            entry = _entity_cache.get(self.session_id)
            if entry:
                return inflate_glossary(entry.state)
        data = self._gql.query(GLOSSARY_QUERY, {"sessionId": self.session_id})
        return data.get("glossary", {}).get("terms", [])

    def glossary_term(self, name: str) -> dict:
        data = self._gql.query(GLOSSARY_TERM_QUERY, {"sessionId": self.session_id, "name": name})
        return data.get("glossary_term", {})

    def agents(self) -> list[dict]:
        data = self._gql.query(AGENTS_QUERY, {"sessionId": self.session_id})
        return data.get("agents", [])

    # ── Binary downloads (stay REST) ────────────────────────────────────

    def table(self, name: str, pandas: bool = False) -> Any:
        resp = self._http.get(
            f"/api/sessions/{self.session_id}/tables/{name}/download",
            params={"format": "parquet"}, timeout=120,
        )
        resp.raise_for_status()
        df = polars.read_parquet(io.BytesIO(resp.content))
        return df.to_pandas() if pandas else df

    def code(self, step: int | None = None) -> str:
        if step is not None:
            codes = self.inference_codes()
            for c in codes:
                if c.get("step_number") == step:
                    return c.get("code", "")
            return ""
        resp = self._http.get(f"/api/sessions/{self.session_id}/download-code")
        resp.raise_for_status()
        return resp.text

    def download_code(self) -> str:
        resp = self._http.get(f"/api/sessions/{self.session_id}/download-inference-code")
        resp.raise_for_status()
        return resp.text

    def export_table(self, name: str, path: str, fmt: str = "csv") -> None:
        import pathlib
        resp = self._http.get(
            f"/api/sessions/{self.session_id}/tables/{name}/download",
            params={"format": fmt}, timeout=120,
        )
        resp.raise_for_status()
        pathlib.Path(path).write_bytes(resp.content)

    # ── Write mutations ─────────────────────────────────────────────────

    def remember(self, name: str, value: str) -> None:
        self._gql.query(ADD_FACT, {"sessionId": self.session_id, "name": name, "value": value})

    def forget(self, name: str) -> None:
        # The GraphQL mutation uses factId (int), but the REST API used name.
        # Find the fact by name, then delete by ID.
        facts = self.facts()
        for f in facts:
            if f.get("name") == name:
                fact_id = f.get("id")
                if fact_id is not None:
                    self._gql.query(FORGET_FACT, {"sessionId": self.session_id, "factId": fact_id})
                    return
        # Fallback: try REST endpoint which accepts name
        resp = self._http.post(f"/api/sessions/{self.session_id}/facts/{name}/forget")
        resp.raise_for_status()

    def set_domains(self, domains: list[str]) -> None:
        self._gql.query(SET_ACTIVE_DOMAINS, {"sessionId": self.session_id, "domains": domains})

    def cancel(self) -> None:
        self._gql.query(CANCEL_EXECUTION, {"sessionId": self.session_id})

    def reset(self) -> None:
        self._gql.query(RESET_CONTEXT, {"sessionId": self.session_id})

    def define(self, name: str, definition: str, **kwargs) -> dict:
        input_data = {"name": name, "definition": definition}
        if kwargs.get("parent_id"):
            input_data["parentId"] = kwargs["parent_id"]
        if kwargs.get("aliases"):
            input_data["aliases"] = kwargs["aliases"]
        if kwargs.get("semantic_type"):
            input_data["semanticType"] = kwargs["semantic_type"]
        data = self._gql.query(CREATE_GLOSSARY_TERM, {"sessionId": self.session_id, "input": input_data})
        return data.get("create_glossary_term", {})

    def undefine(self, name: str) -> None:
        self._gql.query(DELETE_GLOSSARY_TERM, {"sessionId": self.session_id, "name": name})

    def refine(self, name: str) -> dict:
        data = self._gql.query(REFINE_DEFINITION, {"sessionId": self.session_id, "name": name})
        return data.get("refine_definition", {})

    def generate_glossary(self) -> dict:
        data = self._gql.query(GENERATE_GLOSSARY, {"sessionId": self.session_id})
        return data.get("generate_glossary", {})

    def correct(self, text: str) -> dict:
        data = self._gql.query(FLAG_ANSWER, {"sessionId": self.session_id, "text": text})
        return data.get("flag_answer", {})

    def add_database(self, uri: str, name: str | None = None) -> dict:
        input_data: dict[str, Any] = {"name": name or uri.split("/")[-1], "uri": uri}
        data = self._gql.query(ADD_DATABASE, {"sessionId": self.session_id, "input": input_data})
        return data.get("add_database", {})

    def remove_database(self, name: str) -> None:
        self._gql.query(REMOVE_DATABASE, {"sessionId": self.session_id, "name": name})

    def add_api(self, spec_url: str, name: str | None = None) -> dict:
        input_data: dict[str, Any] = {"name": name or spec_url, "baseUrl": spec_url}
        data = self._gql.query(ADD_API, {"sessionId": self.session_id, "input": input_data})
        return data.get("add_api", {})

    def remove_api(self, name: str) -> None:
        self._gql.query(REMOVE_API, {"sessionId": self.session_id, "name": name})

    def add_document(self, uri: str) -> dict:
        input_data = {"name": uri, "url": uri}
        data = self._gql.query(ADD_DOCUMENT_URI, {"sessionId": self.session_id, "input": input_data})
        return data.get("add_document_uri", {})

    def star_table(self, name: str, starred: bool = True) -> None:
        self._gql.query(TOGGLE_TABLE_STAR, {"sessionId": self.session_id, "tableName": name})

    def delete_table(self, name: str) -> None:
        self._gql.query(DELETE_TABLE, {"sessionId": self.session_id, "tableName": name})

    def star_artifact(self, artifact_id: int, starred: bool = True) -> None:
        self._gql.query(TOGGLE_ARTIFACT_STAR, {"sessionId": self.session_id, "artifactId": artifact_id})

    def delete_artifact(self, artifact_id: int) -> None:
        self._gql.query(DELETE_ARTIFACT, {"sessionId": self.session_id, "artifactId": artifact_id})

    # ── File uploads (stay REST — multipart) ────────────────────────────

    def upload_document(self, path: str) -> dict:
        import pathlib
        p = pathlib.Path(path)
        with open(p, "rb") as f:
            resp = self._http.post(
                f"/api/sessions/{self.session_id}/documents/upload",
                files={"file": (p.name, f)},
            )
        resp.raise_for_status()
        return resp.json()

    def files(self) -> list[dict]:
        resp = self._http.get(f"/api/sessions/{self.session_id}/files")
        resp.raise_for_status()
        return resp.json().get("files", [])

    def upload_file(self, path: str) -> dict:
        import pathlib
        p = pathlib.Path(path)
        with open(p, "rb") as f:
            resp = self._http.post(
                f"/api/sessions/{self.session_id}/files",
                files={"file": (p.name, f)},
            )
        resp.raise_for_status()
        return resp.json()

    def delete_file(self, file_id: int) -> None:
        resp = self._http.delete(f"/api/sessions/{self.session_id}/files/{file_id}")
        resp.raise_for_status()

    # ── Agents (GraphQL CRUD) ─────────────────────────────────────────────

    def agent(self, name: str) -> dict:
        resp = self._http.get(f"/api/sessions/{self.session_id}/agents/{name}")
        resp.raise_for_status()
        return resp.json()

    def set_agent(self, name: str) -> None:
        resp = self._http.put(f"/api/sessions/{self.session_id}/agents/current", json={"agent_name": name})
        resp.raise_for_status()

    def create_agent(self, name: str, prompt: str, description: str = "", skills: list[str] | None = None) -> dict:
        """Create a new agent via GraphQL."""
        data = self._gql.query(CREATE_AGENT, {
            "sessionId": self.session_id,
            "input": {"name": name, "prompt": prompt, "description": description, "skills": skills or []},
        })
        return data.get("create_agent", {})

    def edit_agent(self, name: str, prompt: str, description: str = "", skills: list[str] | None = None) -> dict:
        """Update an agent via GraphQL."""
        data = self._gql.query(UPDATE_AGENT, {
            "sessionId": self.session_id,
            "name": name,
            "input": {"prompt": prompt, "description": description, "skills": skills or []},
        })
        return data.get("update_agent", {})

    def delete_agent(self, name: str) -> None:
        """Delete an agent via GraphQL."""
        self._gql.query(DELETE_AGENT, {
            "sessionId": self.session_id, "name": name,
        })

    def draft_agent(self, name: str, description: str) -> dict:
        """Draft an agent via GraphQL."""
        data = self._gql.query(DRAFT_AGENT, {
            "sessionId": self.session_id,
            "input": {"name": name, "userDescription": description},
        })
        return data.get("draft_agent", {})

    # ── Handbook ───────────────────────────────────────────────────────────

    def handbook(self, domain: str | None = None) -> dict:
        """Get the domain handbook for the current session."""
        result = self._gql.query(HANDBOOK_QUERY, {
            "sessionId": self.session_id,
            "domain": domain,
        })
        return result.get("handbook", {})

    # ── Regression testing (stay REST — SSE streaming) ──────────────────

    def test_domains(self) -> list[dict]:
        resp = self._http.get(f"/api/sessions/{self.session_id}/tests/domains")
        resp.raise_for_status()
        return resp.json().get("domains", [])

    def test_questions(self, domain: str) -> list[dict]:
        resp = self._http.get(f"/api/sessions/{self.session_id}/tests/{domain}/questions")
        resp.raise_for_status()
        return resp.json().get("questions", [])

    def create_test_question(self, domain: str, question: dict) -> dict:
        resp = self._http.post(f"/api/sessions/{self.session_id}/tests/{domain}/questions", json=question)
        resp.raise_for_status()
        return resp.json()

    def update_test_question(self, domain: str, index: int, question: dict) -> dict:
        resp = self._http.put(f"/api/sessions/{self.session_id}/tests/{domain}/questions/{index}", json=question)
        resp.raise_for_status()
        return resp.json()

    def delete_test_question(self, domain: str, index: int) -> None:
        resp = self._http.delete(f"/api/sessions/{self.session_id}/tests/{domain}/questions/{index}")
        resp.raise_for_status()

    def run_tests(self, domains: list[str] | None = None, questions: list[int] | None = None) -> list[dict]:
        body: dict[str, Any] = {}
        if domains:
            body["domains"] = domains
        if questions:
            body["question_indices"] = questions
        results = []
        with self._http.stream("POST", f"/api/sessions/{self.session_id}/tests/run", json=body, timeout=600) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    event = data.get("event", "")
                    if event == "question_result":
                        results.append(data.get("data", {}))
                        q = data.get("data", {})
                        status = "PASS" if q.get("passed") else "FAIL"
                        print(f"  [{status}] {q.get('question', '')[:80]}")
                    elif event == "run_complete":
                        summary = data.get("data", {})
                        print(f"\n[done] {summary.get('passed', 0)}/{summary.get('total', 0)} passed")
        return results

    # ── Search (stay REST — no GQL equivalent) ──────────────────────────

    def search_tables(self, query: str) -> dict:
        resp = self._http.get("/api/schema/search", params={"query": query})
        resp.raise_for_status()
        return resp.json()

    def search_apis(self, query: str) -> dict:
        return self.search_tables(query)

    def search_docs(self, query: str) -> dict:
        return self.search_tables(query)

    def search_chunks(self, query: str) -> dict:
        return self.search_tables(query)

    # ── Internal helpers ────────────────────────────────────────────────

    def _table_names(self) -> set[str]:
        try:
            return {t["name"] for t in self.tables()}
        except Exception:
            return set()

    def _artifact_ids(self) -> set[int]:
        try:
            return {a.id for a in self.artifacts()}
        except Exception:
            return set()

    def _fetch_all_tables(self) -> dict[str, polars.DataFrame]:
        try:
            return {t["name"]: self.table(t["name"]) for t in self.tables()}
        except Exception:
            return {}

    def _fetch_new_tables(self, pre_existing: set[str]) -> dict[str, polars.DataFrame]:
        tables = {}
        try:
            for t in self.tables():
                if t["name"] not in pre_existing:
                    try:
                        tables[t["name"]] = self.table(t["name"])
                    except Exception:
                        pass
        except Exception:
            pass
        return tables

    def _fetch_all_artifacts(self) -> list[Artifact]:
        try:
            return [self.artifact(a.id) for a in self.artifacts()]
        except Exception:
            return []

    def _fetch_new_artifacts(self, pre_existing: set[int]) -> list[Artifact]:
        try:
            return [self.artifact(a.id) for a in self.artifacts() if a.id not in pre_existing]
        except Exception:
            return []

    def cached_result(self) -> SolveResult:
        tables = self._fetch_all_tables()
        artifacts = self._fetch_all_artifacts()
        output_data = self.output()
        answer = output_data.get("output", output_data.get("final_answer", ""))
        return SolveResult(
            success=True,
            answer=_strip_trailing_json(answer) if answer else "",
            raw_output=answer,
            tables=tables,
            artifacts=artifacts,
            _session=self,
        )
