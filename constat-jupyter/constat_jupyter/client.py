from __future__ import annotations

import asyncio
import io
import json
import uuid
from typing import Any

import httpx
import polars

from .config import ConstatConfig
from .models import Artifact, ConstatError, SolveResult, StepInfo
from .progress import PrintProgress

try:
    from .widgets import HAS_WIDGETS, widget_plan_approval, widget_clarification, WidgetProgress
except ImportError:
    HAS_WIDGETS = False


def _strip_trailing_json(text: str) -> str:
    """Remove JSON arrays/objects from server output."""
    import json as _json
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
                _json.loads(stripped)
                continue
            except _json.JSONDecodeError:
                pass
        kept.append(line)
    result = '\n'.join(kept)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def _extract_plan(data: dict) -> tuple[str, list[dict]]:
    """Extract problem and steps from plan_ready event data (handles both nested and flat formats)."""
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
            # If user typed a number, map to suggestion
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


class ConstatClient:
    """HTTP client for a running Constat server.

    Usage::

        client = ConstatClient("http://localhost:8000")
        session = client.create_session()
        result = session.solve("What are the top 10 items by value?")
    """

    def __init__(self, server_url: str | None = None, token: str | None = None):
        cfg = ConstatConfig.resolve(server_url, token)
        self._base_url = cfg.server_url.rstrip("/")
        self._token = cfg.token
        headers = {"Authorization": f"Bearer {cfg.token}"} if cfg.token else {}
        self._http = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=30,
        )

    # -- Session management --

    def create_session(self, session_id: str | None = None) -> Session:
        sid = session_id or str(uuid.uuid4())
        resp = self._http.post("/api/sessions", json={"session_id": sid})
        resp.raise_for_status()
        return Session(self, sid)

    def get_session(self, session_id: str) -> Session:
        resp = self._http.get(f"/api/sessions/{session_id}")
        resp.raise_for_status()
        return Session(self, session_id)

    def list_sessions(self) -> list[dict]:
        resp = self._http.get("/api/sessions")
        resp.raise_for_status()
        return resp.json()["sessions"]

    def delete_session(self, session_id: str) -> None:
        resp = self._http.delete(f"/api/sessions/{session_id}")
        resp.raise_for_status()

    # -- Schema browsing --

    def databases(self) -> list[dict]:
        resp = self._http.get("/api/schema")
        resp.raise_for_status()
        return resp.json()["databases"]

    def table_schema(self, database: str, table: str) -> dict:
        resp = self._http.get(f"/api/schema/databases/{database}/tables/{table}")
        resp.raise_for_status()
        return resp.json()

    # -- Domains --

    def domains(self) -> list[dict]:
        resp = self._http.get("/api/domains")
        resp.raise_for_status()
        return resp.json()["domains"]

    # -- Skills --

    def skills(self) -> list[dict]:
        resp = self._http.get("/api/skills")
        resp.raise_for_status()
        return resp.json()["skills"]

    def skill_info(self, name: str) -> dict:
        resp = self._http.get(f"/api/skills/{name}")
        resp.raise_for_status()
        return resp.json()

    def create_skill(self, name: str, content: str) -> dict:
        resp = self._http.post("/api/skills", json={"name": name, "content": content})
        resp.raise_for_status()
        return resp.json()

    def edit_skill(self, name: str, content: str) -> dict:
        resp = self._http.put(f"/api/skills/{name}", json={"content": content})
        resp.raise_for_status()
        return resp.json()

    def delete_skill(self, name: str) -> None:
        resp = self._http.delete(f"/api/skills/{name}")
        resp.raise_for_status()

    def draft_skill(self, name: str, description: str) -> dict:
        resp = self._http.post("/api/skills/draft", json={"name": name, "description": description})
        resp.raise_for_status()
        return resp.json()

    def download_skill(self, name: str) -> bytes:
        resp = self._http.get(f"/api/skills/{name}/download")
        resp.raise_for_status()
        return resp.content

    # -- Schema search --

    def search_schema(self, query: str) -> dict:
        resp = self._http.get("/api/schema/search", params={"query": query})
        resp.raise_for_status()
        return resp.json()

    # -- Learnings & Rules --

    def learnings(self, category: str | None = None) -> list[dict]:
        params = {"category": category} if category else {}
        resp = self._http.get("/api/learnings", params=params)
        resp.raise_for_status()
        return resp.json().get("learnings", [])

    def compact_learnings(self) -> dict:
        resp = self._http.post("/api/learnings/compact")
        resp.raise_for_status()
        return resp.json()

    def add_rule(self, text: str) -> dict:
        resp = self._http.post("/api/rules", json={"text": text})
        resp.raise_for_status()
        return resp.json()

    def edit_rule(self, rule_id: int, text: str) -> dict:
        resp = self._http.put(f"/api/rules/{rule_id}", json={"text": text})
        resp.raise_for_status()
        return resp.json()

    def delete_rule(self, rule_id: int) -> None:
        resp = self._http.delete(f"/api/rules/{rule_id}")
        resp.raise_for_status()

    # -- Lifecycle --

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Session:
    """A Constat session. Created via ``ConstatClient.create_session()``."""

    _css_injected = False

    def __init__(self, client: ConstatClient, session_id: str):
        self._client = client
        self.session_id = session_id
        self._progress = WidgetProgress() if HAS_WIDGETS else PrintProgress()
        self._inject_css()

    @classmethod
    def _inject_css(cls) -> None:
        """Inject global CSS for iTables styling (once per kernel)."""
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

    # -- Query execution --

    async def solve(self, question: str, auto_approve: bool = True, timeout: float = 600) -> SolveResult:
        """Submit a question, stream progress, return result."""
        return await self._execute_async(question, is_followup=False, auto_approve=auto_approve, timeout=timeout)

    async def follow_up(self, question: str, auto_approve: bool = True, timeout: float = 600) -> SolveResult:
        """Follow-up question in the same session context."""
        return await self._execute_async(question, is_followup=True, auto_approve=auto_approve, timeout=timeout)

    async def reason_chain(self, question: str, auto_approve: bool = True, timeout: float = 600) -> SolveResult:
        """Auditable reasoning chain execution."""
        return await self._execute_async(question, is_followup=False, auto_approve=auto_approve, timeout=timeout)

    # -- WebSocket infrastructure --

    def _ws_url(self) -> str:
        ws_url = self._client._base_url.replace("http://", "ws://").replace("https://", "wss://")
        return f"{ws_url}/api/sessions/{self.session_id}/ws"

    def _ws_headers(self) -> dict:
        if self._client._token:
            return {"Authorization": f"Bearer {self._client._token}"}
        return {}

    async def _ws_event_loop(self, ws, auto_approve: bool, timeout: float) -> SolveResult:
        """Shared WS event processing loop."""
        result = SolveResult(success=False, answer="")
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                raise ConstatError(f"Timed out after {timeout}s")

            msg = json.loads(raw)
            if msg.get("type") != "event":
                continue

            payload = msg.get("payload", {})
            event_type = payload.get("event_type", "")
            data = dict(payload.get("data", {}))
            if "step_number" in payload:
                data.setdefault("step_number", payload["step_number"])

            # Server may auto-approve simple plans — check before prompting
            server_auto_approved = (event_type == "plan_ready" and data.get("auto_approved"))
            needs_client_approval = not auto_approve and not server_auto_approved

            # Skip progress display for events handled by interactive widgets
            if HAS_WIDGETS and needs_client_approval and event_type in ("planning_start", "clarification_needed"):
                pass  # Widget handles display
            else:
                self._progress.handle_event(event_type, data)

            if event_type == "plan_ready":
                if auto_approve or server_auto_approved:
                    if not server_auto_approved:
                        await ws.send(json.dumps({"action": "approve"}))
                else:
                    if HAS_WIDGETS:
                        decision = await widget_plan_approval(data)
                    else:
                        loop = asyncio.get_event_loop()
                        decision = await loop.run_in_executor(None, _prompt_plan_approval, data)
                    if decision.get("approved"):
                        approve_data: dict[str, Any] = {}
                        if decision.get("deleted_steps"):
                            approve_data["deleted_steps"] = decision["deleted_steps"]
                        if decision.get("edited_steps"):
                            approve_data["edited_steps"] = decision["edited_steps"]
                        msg = {"action": "approve"}
                        if approve_data:
                            msg["data"] = approve_data
                        await ws.send(json.dumps(msg))
                    else:
                        await ws.send(json.dumps({"action": "reject", "data": {"feedback": decision.get("feedback", "")}}))

            elif event_type == "clarification_needed":
                if HAS_WIDGETS:
                    answers = await widget_clarification(data)
                else:
                    loop = asyncio.get_event_loop()
                    answers = await loop.run_in_executor(None, _prompt_clarifications, data)
                if answers:
                    await ws.send(json.dumps({"action": "clarify", "data": {"answers": answers}}))
                else:
                    await ws.send(json.dumps({"action": "skip_clarification"}))

            elif event_type == "query_complete":
                result.success = True
                raw_output = data.get("output", "")
                result.raw_output = raw_output
                result.answer = _strip_trailing_json(raw_output)
                result.suggestions = data.get("suggestions", [])
                break

            elif event_type == "query_error":
                result.error = data.get("error", "Unknown error")
                break

            elif event_type == "query_cancelled":
                result.error = "Query was cancelled"
                break

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

        return result

    async def _execute_async(
        self,
        question: str,
        is_followup: bool,
        auto_approve: bool,
        timeout: float,
    ) -> SolveResult:
        import websockets

        async with websockets.connect(self._ws_url(), additional_headers=self._ws_headers()) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            body: dict[str, Any] = {"problem": question, "is_followup": is_followup}
            if not auto_approve:
                body["require_approval"] = True
            resp = self._client._http.post(
                f"/api/sessions/{self.session_id}/query",
                json=body,
            )
            resp.raise_for_status()

            result = await self._ws_event_loop(ws, auto_approve, timeout)

        if result.success:
            result.tables = self._fetch_all_tables()
            result._session = self

        return result

    async def _ws_action_async(
        self,
        action: str,
        data: dict | None = None,
        auto_approve: bool = True,
        timeout: float = 300,
    ) -> SolveResult:
        """Send a WebSocket action (replan_from, edit_objective, etc.) and stream events."""
        import websockets

        async with websockets.connect(self._ws_url(), additional_headers=self._ws_headers()) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            msg: dict[str, Any] = {"action": action}
            if data:
                msg["data"] = data
            await ws.send(json.dumps(msg))

            result = await self._ws_event_loop(ws, auto_approve, timeout)

        if result.success:
            result.tables = self._fetch_all_tables()
            result._session = self

        return result

    def _fetch_all_tables(self) -> dict[str, polars.DataFrame]:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/tables")
        if not resp.is_success:
            return {}
        tables = {}
        for t in resp.json().get("tables", []):
            name = t["name"]
            try:
                tables[name] = self.table(name)
            except Exception:
                pass  # Skip tables that can't be downloaded
        return tables

    # -- Table access --

    def table(self, name: str, pandas: bool = False) -> Any:
        """Download a session table as a Polars (default) or Pandas DataFrame."""
        resp = self._client._http.get(
            f"/api/sessions/{self.session_id}/tables/{name}/download",
            params={"format": "parquet"},
            timeout=120,
        )
        resp.raise_for_status()
        df = polars.read_parquet(io.BytesIO(resp.content))
        if pandas:
            return df.to_pandas()
        return df

    def tables(self) -> list[dict]:
        """List all tables in the session."""
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/tables")
        resp.raise_for_status()
        return resp.json()["tables"]

    # -- Artifacts --

    def artifacts(self) -> list[Artifact]:
        """List all artifacts in the session."""
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/artifacts")
        resp.raise_for_status()
        return [
            Artifact(
                id=a["id"],
                name=a["name"],
                artifact_type=a["artifact_type"],
                mime_type=a.get("mime_type"),
                is_starred=a.get("is_starred", False),
            )
            for a in resp.json().get("artifacts", [])
        ]

    def artifact(self, artifact_id: int) -> Artifact:
        """Fetch a single artifact with content."""
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/artifacts/{artifact_id}")
        resp.raise_for_status()
        a = resp.json()
        return Artifact(
            id=a["id"],
            name=a["name"],
            artifact_type=a["artifact_type"],
            content=a.get("content"),
            mime_type=a.get("mime_type"),
        )

    # -- Data sources --

    def sources(self) -> dict:
        """List all data sources: databases, APIs, documents."""
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/sources")
        resp.raise_for_status()
        return resp.json()

    def databases(self) -> list[dict]:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/databases")
        resp.raise_for_status()
        return resp.json().get("databases", [])

    # -- Glossary --

    def glossary(self) -> list[dict]:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/glossary")
        resp.raise_for_status()
        return resp.json().get("terms", [])

    def glossary_term(self, name: str) -> dict:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/glossary/{name}")
        resp.raise_for_status()
        return resp.json()

    # -- Facts --

    def facts(self) -> list[dict]:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/facts")
        resp.raise_for_status()
        return resp.json().get("facts", [])

    def remember(self, name: str, value: str) -> None:
        resp = self._client._http.post(
            f"/api/sessions/{self.session_id}/facts",
            json={"name": name, "value": value},
        )
        resp.raise_for_status()

    def forget(self, name: str) -> None:
        resp = self._client._http.post(f"/api/sessions/{self.session_id}/facts/{name}/forget")
        resp.raise_for_status()

    # -- Session control --

    def set_domains(self, domains: list[str]) -> None:
        resp = self._client._http.post(
            f"/api/sessions/{self.session_id}/domains",
            json={"domains": domains},
        )
        resp.raise_for_status()

    def cancel(self) -> None:
        resp = self._client._http.post(f"/api/sessions/{self.session_id}/cancel")
        resp.raise_for_status()

    # -- Commands (sent as /slash query text, processed by backend command dispatcher) --

    async def command(self, cmd: str, timeout: float = 120) -> SolveResult:
        """Send a REPL slash command (e.g. '/compact', '/summarize plan').

        Commands are sent as query text via POST /query. The backend command
        dispatcher intercepts queries starting with '/' before the normal
        planning/execution pipeline.
        """
        return await self._execute_async(cmd, is_followup=False, auto_approve=True, timeout=timeout)

    async def compact(self) -> SolveResult:
        """Compact context to reduce token usage."""
        return await self.command("/compact")

    async def save_plan(self, name: str) -> SolveResult:
        """Save current plan for replay."""
        return await self.command(f"/save {name}")

    async def share_plan(self, name: str) -> SolveResult:
        """Save plan as shared (all users)."""
        return await self.command(f"/share {name}")

    async def list_plans(self) -> SolveResult:
        """List saved plans."""
        return await self.command("/plans")

    async def replay_plan(self, name: str, auto_approve: bool = True, timeout: float = 600) -> SolveResult:
        """Replay a saved plan."""
        return await self._execute_async(f"/replay {name}", is_followup=False, auto_approve=auto_approve, timeout=timeout)

    async def summarize(self, target: str) -> SolveResult:
        """Summarize plan|session|facts|<table_name>."""
        return await self.command(f"/summarize {target}")

    async def discover(self, query: str) -> SolveResult:
        """Search all data sources (tables, APIs, documents, glossary)."""
        return await self.command(f"/discover {query}")

    async def redo(self, instruction: str | None = None, auto_approve: bool = True, timeout: float = 600) -> SolveResult:
        """Re-run last query, optionally with modified instruction."""
        cmd = f"/redo {instruction}" if instruction else "/redo"
        return await self._execute_async(cmd, is_followup=False, auto_approve=auto_approve, timeout=timeout)

    async def audit(self) -> SolveResult:
        """Audit last result."""
        return await self.command("/audit")

    async def set_verbose(self, on: bool | None = None) -> SolveResult:
        """Toggle verbose mode."""
        arg = " on" if on is True else " off" if on is False else ""
        return await self.command(f"/verbose{arg}")

    async def set_raw(self, on: bool | None = None) -> SolveResult:
        """Toggle raw output display."""
        arg = " on" if on is True else " off" if on is False else ""
        return await self.command(f"/raw{arg}")

    async def set_insights(self, on: bool | None = None) -> SolveResult:
        """Toggle insight synthesis."""
        arg = " on" if on is True else " off" if on is False else ""
        return await self.command(f"/insights{arg}")

    async def preferences(self) -> SolveResult:
        """Show current preferences."""
        return await self.command("/preferences")

    async def objectives(self) -> SolveResult:
        """List session objectives."""
        return await self.command("/objectives")

    # -- WS actions (step & objective manipulation, triggers replanning) --

    async def step_edit(self, step_number: int, goal: str, auto_approve: bool = True) -> SolveResult:
        """Edit a plan step goal and replan from that step."""
        return await self._ws_action_async(
            "replan_from",
            {"step_number": step_number, "mode": "edit", "edited_goal": goal},
            auto_approve=auto_approve,
        )

    async def step_delete(self, step_number: int, auto_approve: bool = True) -> SolveResult:
        """Delete a plan step and replan."""
        return await self._ws_action_async(
            "replan_from",
            {"step_number": step_number, "mode": "delete"},
            auto_approve=auto_approve,
        )

    async def step_redo(self, step_number: int, auto_approve: bool = True) -> SolveResult:
        """Re-execute from a specific plan step."""
        return await self._ws_action_async(
            "replan_from",
            {"step_number": step_number, "mode": "redo"},
            auto_approve=auto_approve,
        )

    async def edit_objective(self, index: int, text: str, auto_approve: bool = True) -> SolveResult:
        """Edit an objective and replan from first affected step."""
        return await self._ws_action_async(
            "edit_objective",
            {"objective_index": index, "new_text": text},
            auto_approve=auto_approve,
        )

    async def delete_objective(self, index: int, auto_approve: bool = True) -> SolveResult:
        """Delete an objective and replan."""
        return await self._ws_action_async(
            "delete_objective",
            {"objective_index": index},
            auto_approve=auto_approve,
        )

    @property
    def status(self) -> dict:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}")
        resp.raise_for_status()
        return resp.json()

    # -- Session management --

    def reset(self) -> None:
        resp = self._client._http.post(f"/api/sessions/{self.session_id}/reset-context")
        resp.raise_for_status()

    def context(self) -> dict:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/prompt-context")
        resp.raise_for_status()
        return resp.json()

    # -- Plan & Steps --

    def plan(self) -> dict:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/plan")
        resp.raise_for_status()
        return resp.json()

    def steps(self) -> list[dict]:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/steps")
        resp.raise_for_status()
        return resp.json().get("steps", [])

    def code(self, step: int | None = None) -> str:
        if step is not None:
            codes = self.inference_codes()
            for c in codes:
                if c.get("step_number") == step:
                    return c.get("code", "")
            return ""
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/download-code")
        resp.raise_for_status()
        return resp.text

    def inference_codes(self) -> list[dict]:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/inference-codes")
        resp.raise_for_status()
        return resp.json().get("codes", [])

    def download_code(self) -> str:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/download-inference-code")
        resp.raise_for_status()
        return resp.text

    # -- DDL & SQL --

    def ddl(self) -> str:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/ddl")
        resp.raise_for_status()
        return resp.json().get("ddl", "")

    # -- Diagnostics --

    def search_tables(self, query: str) -> dict:
        return self._client.search_schema(query)

    def search_apis(self, query: str) -> dict:
        return self._client.search_schema(query)

    def search_docs(self, query: str) -> dict:
        return self._client.search_schema(query)

    def search_chunks(self, query: str) -> dict:
        return self._client.search_schema(query)

    def entities(self) -> list[dict]:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/entities")
        resp.raise_for_status()
        return resp.json().get("entities", [])

    def proof_tree(self) -> dict:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/proof-tree")
        resp.raise_for_status()
        return resp.json()

    def output(self) -> dict:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/output")
        resp.raise_for_status()
        return resp.json()

    def scratchpad(self) -> dict:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/scratchpad")
        resp.raise_for_status()
        return resp.json()

    # -- Glossary (write operations) --

    def define(self, name: str, definition: str, **kwargs) -> dict:
        body = {"name": name, "definition": definition, **kwargs}
        resp = self._client._http.post(f"/api/sessions/{self.session_id}/glossary", json=body)
        resp.raise_for_status()
        return resp.json()

    def undefine(self, name: str) -> None:
        resp = self._client._http.delete(f"/api/sessions/{self.session_id}/glossary/{name}")
        resp.raise_for_status()

    def refine(self, name: str) -> dict:
        resp = self._client._http.post(f"/api/sessions/{self.session_id}/glossary/{name}/refine")
        resp.raise_for_status()
        return resp.json()

    def generate_glossary(self) -> dict:
        resp = self._client._http.post(f"/api/sessions/{self.session_id}/glossary/generate")
        resp.raise_for_status()
        return resp.json()

    # -- Facts (extended) --

    def correct(self, text: str) -> dict:
        resp = self._client._http.post(
            f"/api/sessions/{self.session_id}/feedback/flag",
            json={"text": text},
        )
        resp.raise_for_status()
        return resp.json()

    # -- Data sources (write) --

    def add_database(self, uri: str, name: str | None = None) -> dict:
        body = {"uri": uri}
        if name:
            body["name"] = name
        resp = self._client._http.post(f"/api/sessions/{self.session_id}/databases", json=body)
        resp.raise_for_status()
        return resp.json()

    def remove_database(self, name: str) -> None:
        resp = self._client._http.delete(f"/api/sessions/{self.session_id}/databases/{name}")
        resp.raise_for_status()

    def add_api(self, spec_url: str, name: str | None = None) -> dict:
        body = {"spec_url": spec_url}
        if name:
            body["name"] = name
        resp = self._client._http.post(f"/api/sessions/{self.session_id}/apis", json=body)
        resp.raise_for_status()
        return resp.json()

    def remove_api(self, name: str) -> None:
        resp = self._client._http.delete(f"/api/sessions/{self.session_id}/apis/{name}")
        resp.raise_for_status()

    def add_document(self, uri: str) -> dict:
        resp = self._client._http.post(
            f"/api/sessions/{self.session_id}/documents/add-uri",
            json={"uri": uri},
        )
        resp.raise_for_status()
        return resp.json()

    def upload_document(self, path: str) -> dict:
        import pathlib
        p = pathlib.Path(path)
        with open(p, "rb") as f:
            resp = self._client._http.post(
                f"/api/sessions/{self.session_id}/documents/upload",
                files={"file": (p.name, f)},
            )
        resp.raise_for_status()
        return resp.json()

    def files(self) -> list[dict]:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/files")
        resp.raise_for_status()
        return resp.json().get("files", [])

    def upload_file(self, path: str) -> dict:
        import pathlib
        p = pathlib.Path(path)
        with open(p, "rb") as f:
            resp = self._client._http.post(
                f"/api/sessions/{self.session_id}/files",
                files={"file": (p.name, f)},
            )
        resp.raise_for_status()
        return resp.json()

    def delete_file(self, file_id: int) -> None:
        resp = self._client._http.delete(f"/api/sessions/{self.session_id}/files/{file_id}")
        resp.raise_for_status()

    # -- Agents --

    def agents(self) -> list[dict]:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/agents")
        resp.raise_for_status()
        return resp.json().get("agents", [])

    def agent(self, name: str) -> dict:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/agents/{name}")
        resp.raise_for_status()
        return resp.json()

    def set_agent(self, name: str) -> None:
        resp = self._client._http.put(
            f"/api/sessions/{self.session_id}/agents/current",
            json={"agent_name": name},
        )
        resp.raise_for_status()

    def create_agent(self, name: str, content: str) -> dict:
        resp = self._client._http.post(
            f"/api/sessions/{self.session_id}/agents",
            json={"name": name, "content": content},
        )
        resp.raise_for_status()
        return resp.json()

    def edit_agent(self, name: str, content: str) -> dict:
        resp = self._client._http.put(
            f"/api/sessions/{self.session_id}/agents/{name}",
            json={"content": content},
        )
        resp.raise_for_status()
        return resp.json()

    def delete_agent(self, name: str) -> None:
        resp = self._client._http.delete(f"/api/sessions/{self.session_id}/agents/{name}")
        resp.raise_for_status()

    def draft_agent(self, name: str, description: str) -> dict:
        resp = self._client._http.post(
            f"/api/sessions/{self.session_id}/agents/draft",
            json={"name": name, "description": description},
        )
        resp.raise_for_status()
        return resp.json()

    # -- Regression testing --

    def test_domains(self) -> list[dict]:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/tests/domains")
        resp.raise_for_status()
        return resp.json().get("domains", [])

    def test_questions(self, domain: str) -> list[dict]:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/tests/{domain}/questions")
        resp.raise_for_status()
        return resp.json().get("questions", [])

    def create_test_question(self, domain: str, question: dict) -> dict:
        resp = self._client._http.post(
            f"/api/sessions/{self.session_id}/tests/{domain}/questions",
            json=question,
        )
        resp.raise_for_status()
        return resp.json()

    def update_test_question(self, domain: str, index: int, question: dict) -> dict:
        resp = self._client._http.put(
            f"/api/sessions/{self.session_id}/tests/{domain}/questions/{index}",
            json=question,
        )
        resp.raise_for_status()
        return resp.json()

    def delete_test_question(self, domain: str, index: int) -> None:
        resp = self._client._http.delete(
            f"/api/sessions/{self.session_id}/tests/{domain}/questions/{index}",
        )
        resp.raise_for_status()

    def run_tests(self, domains: list[str] | None = None, questions: list[int] | None = None) -> list[dict]:
        body: dict[str, Any] = {}
        if domains:
            body["domains"] = domains
        if questions:
            body["question_indices"] = questions
        results = []
        with self._client._http.stream(
            "POST",
            f"/api/sessions/{self.session_id}/tests/run",
            json=body,
            timeout=600,
        ) as resp:
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

    # -- Table operations --

    def star_table(self, name: str, starred: bool = True) -> None:
        resp = self._client._http.post(
            f"/api/sessions/{self.session_id}/tables/{name}/star",
            json={"starred": starred},
        )
        resp.raise_for_status()

    def delete_table(self, name: str) -> None:
        resp = self._client._http.delete(f"/api/sessions/{self.session_id}/tables/{name}")
        resp.raise_for_status()

    def export_table(self, name: str, path: str, fmt: str = "csv") -> None:
        import pathlib
        resp = self._client._http.get(
            f"/api/sessions/{self.session_id}/tables/{name}/download",
            params={"format": fmt},
            timeout=120,
        )
        resp.raise_for_status()
        pathlib.Path(path).write_bytes(resp.content)

    # -- Artifact operations --

    def star_artifact(self, artifact_id: int, starred: bool = True) -> None:
        resp = self._client._http.post(
            f"/api/sessions/{self.session_id}/artifacts/{artifact_id}/star",
            json={"starred": starred},
        )
        resp.raise_for_status()

    def delete_artifact(self, artifact_id: int) -> None:
        resp = self._client._http.delete(f"/api/sessions/{self.session_id}/artifacts/{artifact_id}")
        resp.raise_for_status()

    # -- Messages --

    def messages(self) -> list[dict]:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}/messages")
        resp.raise_for_status()
        return resp.json().get("messages", [])
