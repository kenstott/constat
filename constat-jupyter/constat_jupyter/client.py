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


def _run_async(coro):
    """Run async coroutine in a dedicated thread to avoid Jupyter event loop conflicts."""
    import threading

    result = [None]
    error = [None]

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result[0] = loop.run_until_complete(coro)
        except Exception as e:
            error[0] = e
        finally:
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join()

    if error[0] is not None:
        raise error[0]
    return result[0]


def _strip_trailing_json(text: str) -> str:
    """Remove trailing JSON arrays/objects from server output."""
    lines = text.split('\n')
    while lines and lines[-1].strip() == '':
        lines.pop()
    while lines and (lines[-1].strip().startswith('[{') or lines[-1].strip().startswith('{"')):
        lines.pop()
        while lines and lines[-1].strip() == '':
            lines.pop()
    return '\n'.join(lines).rstrip()


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

    # -- Lifecycle --

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Session:
    """A Constat session. Created via ``ConstatClient.create_session()``."""

    def __init__(self, client: ConstatClient, session_id: str):
        self._client = client
        self.session_id = session_id
        self._progress = PrintProgress()

    # -- Query execution --

    def solve(self, question: str, auto_approve: bool = True, timeout: float = 600) -> SolveResult:
        """Submit a question, stream progress, return result."""
        return _run_async(self._execute_async(question, is_followup=False, auto_approve=auto_approve, timeout=timeout))

    def follow_up(self, question: str, auto_approve: bool = True, timeout: float = 600) -> SolveResult:
        """Follow-up question in the same session context."""
        return _run_async(self._execute_async(question, is_followup=True, auto_approve=auto_approve, timeout=timeout))

    def reason_chain(self, question: str, auto_approve: bool = True, timeout: float = 600) -> SolveResult:
        """Auditable reasoning chain execution."""
        return _run_async(self._execute_async(question, is_followup=False, auto_approve=auto_approve, timeout=timeout))

    async def _execute_async(
        self,
        question: str,
        is_followup: bool,
        auto_approve: bool,
        timeout: float,
    ) -> SolveResult:
        import websockets

        http = self._client._http
        sid = self.session_id

        ws_url = self._client._base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/api/sessions/{sid}/ws"
        extra_headers = {}
        if self._client._token:
            extra_headers["Authorization"] = f"Bearer {self._client._token}"

        result = SolveResult(success=False, answer="")

        async with websockets.connect(ws_url, additional_headers=extra_headers) as ws:
            # Wait for welcome
            await asyncio.wait_for(ws.recv(), timeout=10)

            # Start query
            resp = http.post(
                f"/api/sessions/{sid}/query",
                json={"problem": question, "is_followup": is_followup},
            )
            resp.raise_for_status()

            # Event loop
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    raise ConstatError(f"Query timed out after {timeout}s")

                msg = json.loads(raw)
                if msg.get("type") != "event":
                    continue

                payload = msg.get("payload", {})
                event_type = payload.get("event_type", "")
                data = dict(payload.get("data", {}))
                if "step_number" in payload:
                    data.setdefault("step_number", payload["step_number"])

                self._progress.handle_event(event_type, data)

                if event_type == "plan_ready":
                    if auto_approve:
                        await ws.send(json.dumps({"action": "approve"}))
                    else:
                        decision = _prompt_plan_approval(data)
                        if decision.get("approved"):
                            await ws.send(json.dumps({"action": "approve"}))
                        else:
                            await ws.send(json.dumps({"action": "reject", "data": {"feedback": decision.get("feedback", "")}}))

                elif event_type == "clarification_needed":
                    answers = _prompt_clarifications(data)
                    if answers:
                        await ws.send(json.dumps({"action": "clarify", "data": {"answers": answers}}))
                    else:
                        await ws.send(json.dumps({"action": "skip_clarification"}))

                elif event_type == "query_complete":
                    result.success = True
                    raw = data.get("output", "")
                    result.raw_output = raw
                    result.answer = _strip_trailing_json(raw)
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

        # Fetch tables after WS closes
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

    @property
    def status(self) -> dict:
        resp = self._client._http.get(f"/api/sessions/{self.session_id}")
        resp.raise_for_status()
        return resp.json()
