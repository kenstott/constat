# Copyright (c) 2025 Kenneth Stott
# Canary: 5ae37db1-c8c4-4600-a666-7b01621356d8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""ConstatClient — GraphQL-based client for a Constat server."""

from __future__ import annotations

import uuid

import httpx

from .config import ConstatConfig
from .graphql import (
    GraphQLClient,
    CREATE_SESSION, DELETE_SESSION, SESSIONS_QUERY, SESSION_QUERY,
    DOMAINS_QUERY, SKILLS_QUERY, SKILL_QUERY, LEARNINGS_QUERY,
    COMPACT_LEARNINGS, CREATE_RULE, UPDATE_RULE, DELETE_RULE,
)
from .session import Session


class ConstatClient:
    """HTTP client for a running Constat server.

    Usage::

        client = ConstatClient("http://localhost:8000")
        session = client.create_session()
        result = await session.solve("What are the top 10 items by value?")
    """

    def __init__(self, server_url: str | None = None, token: str | None = None):
        cfg = ConstatConfig.resolve(server_url, token)
        self._base_url = cfg.server_url.rstrip("/")
        self._token = cfg.token
        headers = {"Authorization": f"Bearer {cfg.token}"} if cfg.token else {}
        self._http = httpx.Client(base_url=self._base_url, headers=headers, timeout=30)
        self._gql = GraphQLClient(self._base_url, cfg.token)

    # ── Session management ──────────────────────────────────────────────

    def create_session(self, session_id: str | None = None) -> Session:
        sid = session_id or str(uuid.uuid4())
        self._gql.query(CREATE_SESSION, {"sessionId": sid})
        return Session(self._gql, self._http, sid)

    def get_session(self, session_id: str) -> Session:
        self._gql.query(SESSION_QUERY, {"sessionId": session_id})
        return Session(self._gql, self._http, session_id)

    def list_sessions(self) -> list[dict]:
        data = self._gql.query(SESSIONS_QUERY)
        return data.get("sessions", {}).get("sessions", [])

    def delete_session(self, session_id: str) -> None:
        self._gql.query(DELETE_SESSION, {"sessionId": session_id})

    # ── Schema browsing (REST — no GQL equivalent for /api/schema) ──────

    def databases(self) -> list[dict]:
        resp = self._http.get("/api/schema")
        resp.raise_for_status()
        return resp.json()["databases"]

    def table_schema(self, database: str, table: str) -> dict:
        resp = self._http.get(f"/api/schema/databases/{database}/tables/{table}")
        resp.raise_for_status()
        return resp.json()

    # ── Domains ─────────────────────────────────────────────────────────

    def domains(self) -> list[dict]:
        data = self._gql.query(DOMAINS_QUERY)
        return data.get("domains", {}).get("domains", [])

    # ── Skills ──────────────────────────────────────────────────────────

    def skills(self) -> list[dict]:
        data = self._gql.query(SKILLS_QUERY)
        return data.get("skills", {}).get("skills", [])

    def skill_info(self, name: str) -> dict:
        data = self._gql.query(SKILL_QUERY, {"name": name})
        return data.get("skill", {})

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

    # ── Schema search (REST — no GQL equivalent) ───────────────────────

    def search_schema(self, query: str) -> dict:
        resp = self._http.get("/api/schema/search", params={"query": query})
        resp.raise_for_status()
        return resp.json()

    # ── Learnings & Rules ──────────────────────────────────────────────

    def learnings(self, category: str | None = None) -> list[dict]:
        data = self._gql.query(LEARNINGS_QUERY, {"category": category})
        return data.get("learnings", {}).get("learnings", [])

    def compact_learnings(self) -> dict:
        data = self._gql.query(COMPACT_LEARNINGS)
        return data.get("compact_learnings", {})

    def add_rule(self, text: str) -> dict:
        data = self._gql.query(CREATE_RULE, {"text": text})
        return data.get("create_rule", {})

    def edit_rule(self, rule_id: int, text: str) -> dict:
        data = self._gql.query(UPDATE_RULE, {"ruleId": rule_id, "text": text})
        return data.get("update_rule", {})

    def delete_rule(self, rule_id: int) -> None:
        self._gql.query(DELETE_RULE, {"ruleId": rule_id})

    # ── Lifecycle ──────────────────────────────────────────────────────

    def close(self) -> None:
        self._gql.close()
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
