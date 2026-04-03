# Copyright (c) 2025 Kenneth Stott
# Canary: e004e57e-60a6-47e1-8b91-8501ce55ae34
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL-over-HTTP client and operation constants for constat-jupyter.

Provides:
- ``GraphQLClient`` — sync queries/mutations + async SSE subscriptions
- Operation string constants for all GraphQL operations
- ``camel_to_snake`` key converter (GraphQL returns camelCase)
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator

import httpx

from .models import ConstatError

# ---------------------------------------------------------------------------
# camelCase → snake_case converter
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])([A-Z])")


def camel_to_snake(obj: Any) -> Any:
    """Recursively convert camelCase dict keys to snake_case."""
    if isinstance(obj, dict):
        return {_CAMEL_RE.sub(r"_\1", k).lower(): camel_to_snake(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [camel_to_snake(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# GraphQL client
# ---------------------------------------------------------------------------


class GraphQLClient:
    """Thin GraphQL-over-HTTP client using httpx."""

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 30) -> None:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._base_url = base_url.rstrip("/")
        self._headers = headers
        self._timeout = timeout
        self._http = httpx.Client(
            base_url=self._base_url, headers=headers, timeout=timeout,
        )
        self._async_http = httpx.AsyncClient(
            base_url=self._base_url, headers=headers, timeout=timeout,
        )

    def query(self, operation: str, variables: dict | None = None) -> dict:
        """Synchronous GraphQL query or mutation. Returns the ``data`` dict."""
        resp = self._http.post(
            "/api/graphql",
            json={"query": operation, "variables": variables or {}},
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("errors"):
            raise ConstatError(body["errors"][0]["message"])
        return camel_to_snake(body["data"])

    async def query_async(self, operation: str, variables: dict | None = None) -> dict:
        """Async GraphQL query or mutation. Returns the ``data`` dict."""
        resp = await self._async_http.post(
            "/api/graphql",
            json={"query": operation, "variables": variables or {}},
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("errors"):
            raise ConstatError(body["errors"][0]["message"])
        return camel_to_snake(body["data"])

    async def subscribe_sse(
        self, operation: str, variables: dict | None = None, timeout: float = 600,
    ) -> AsyncGenerator[dict, None]:
        """Subscribe to a GraphQL subscription via SSE.

        Yields parsed event ``data`` dicts (already snake_case-converted).
        """
        params = {
            "query": operation,
            "variables": json.dumps(variables or {}),
        }
        async with httpx.AsyncClient(
            base_url=self._base_url, headers=self._headers, timeout=timeout,
        ) as client:
            async with client.stream(
                "GET", "/api/graphql/stream", params=params,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        payload = line[6:]
                        if not payload:
                            return  # complete event
                        event = json.loads(payload)
                        yield camel_to_snake(event.get("data", {}))
                    elif line.startswith("event: complete"):
                        return

    def close(self) -> None:
        self._http.close()


# ---------------------------------------------------------------------------
# Subscription
# ---------------------------------------------------------------------------

QUERY_EXECUTION_SUBSCRIPTION = """
subscription QueryExecution($sessionId: String!) {
  queryExecution(sessionId: $sessionId) {
    eventType
    sessionId
    stepNumber
    timestamp
    data
  }
}
"""

# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

CREATE_SESSION = """
mutation CreateSession($sessionId: String!, $userId: String) {
  createSession(sessionId: $sessionId, userId: $userId) {
    sessionId
    status
    activeDomains
  }
}
"""

DELETE_SESSION = """
mutation DeleteSession($sessionId: String!) {
  deleteSession(sessionId: $sessionId)
}
"""

SESSIONS_QUERY = """
query Sessions {
  sessions { sessions { sessionId status activeDomains createdAt } }
}
"""

SESSION_QUERY = """
query Session($sessionId: String!) {
  session(sessionId: $sessionId) {
    sessionId status activeDomains createdAt
  }
}
"""

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

SUBMIT_QUERY = """
mutation SubmitQuery($sessionId: String!, $input: SubmitQueryInput!) {
  submitQuery(sessionId: $sessionId, input: $input) {
    executionId status message
  }
}
"""

CANCEL_EXECUTION = """
mutation CancelExecution($sessionId: String!) {
  cancelExecution(sessionId: $sessionId) { status message }
}
"""

APPROVE_PLAN = """
mutation ApprovePlan($sessionId: String!, $input: ApprovePlanInput!) {
  approvePlan(sessionId: $sessionId, input: $input) { status message }
}
"""

ANSWER_CLARIFICATION = """
mutation AnswerClarification($sessionId: String!, $answers: JSON!, $structuredAnswers: JSON!) {
  answerClarification(sessionId: $sessionId, answers: $answers, structuredAnswers: $structuredAnswers) {
    status message
  }
}
"""

SKIP_CLARIFICATION = """
mutation SkipClarification($sessionId: String!) {
  skipClarification(sessionId: $sessionId) { status message }
}
"""

REPLAN_FROM = """
mutation ReplanFrom($sessionId: String!, $stepNumber: Int!, $mode: String!, $editedGoal: String) {
  replanFrom(sessionId: $sessionId, stepNumber: $stepNumber, mode: $mode, editedGoal: $editedGoal) {
    status message
  }
}
"""

EDIT_OBJECTIVE = """
mutation EditObjective($sessionId: String!, $objectiveIndex: Int!, $newText: String!) {
  editObjective(sessionId: $sessionId, objectiveIndex: $objectiveIndex, newText: $newText) {
    status message
  }
}
"""

DELETE_OBJECTIVE = """
mutation DeleteObjective($sessionId: String!, $objectiveIndex: Int!) {
  deleteObjective(sessionId: $sessionId, objectiveIndex: $objectiveIndex) {
    status message
  }
}
"""

EXECUTION_PLAN_QUERY = """
query ExecutionPlan($sessionId: String!) {
  executionPlan(sessionId: $sessionId) {
    problem
    steps { number goal status }
    currentStep completedSteps failedSteps isComplete
  }
}
"""

# ---------------------------------------------------------------------------
# Read-only session state
# ---------------------------------------------------------------------------

TABLES_QUERY = """
query Tables($sessionId: String!) {
  tables(sessionId: $sessionId) {
    tables {
      name rowCount isStarred version columns
    }
    total
  }
}
"""

TABLE_DATA_QUERY = """
query TableData($sessionId: String!, $tableName: String!, $page: Int, $pageSize: Int) {
  tableData(sessionId: $sessionId, tableName: $tableName, page: $page, pageSize: $pageSize) {
    name columns data totalRows page pageSize hasMore
  }
}
"""

ARTIFACTS_QUERY = """
query Artifacts($sessionId: String!) {
  artifacts(sessionId: $sessionId) {
    artifacts {
      id name artifactType mimeType isStarred version stepNumber
    }
    total
  }
}
"""

ARTIFACT_QUERY = """
query Artifact($sessionId: String!, $artifactId: Int!) {
  artifact(sessionId: $sessionId, artifactId: $artifactId) {
    id name artifactType content mimeType isStarred version stepNumber
  }
}
"""

FACTS_QUERY = """
query Facts($sessionId: String!) {
  facts(sessionId: $sessionId) {
    facts { name value source isPersisted tags }
    total
  }
}
"""

ENTITIES_QUERY = """
query Entities($sessionId: String!) {
  entities(sessionId: $sessionId) {
    entities { id name displayName semanticType nerType domainId }
    total
  }
}
"""

STEPS_QUERY = """
query Steps($sessionId: String!) {
  steps(sessionId: $sessionId) {
    steps { stepNumber goal code model prompt }
  }
}
"""

INFERENCE_CODES_QUERY = """
query InferenceCodes($sessionId: String!) {
  inferenceCodes(sessionId: $sessionId) {
    codes { inferenceId name operation code attempt prompt model }
  }
}
"""

SCRATCHPAD_QUERY = """
query Scratchpad($sessionId: String!) {
  scratchpad(sessionId: $sessionId) {
    entries { stepNumber goal narrative tablesCreated code userQuery objectiveIndex }
  }
}
"""

SESSION_DDL_QUERY = """
query SessionDDL($sessionId: String!) {
  sessionDdl(sessionId: $sessionId)
}
"""

EXECUTION_OUTPUT_QUERY = """
query ExecutionOutput($sessionId: String!) {
  executionOutput(sessionId: $sessionId) {
    output suggestions
  }
}
"""

PROOF_TREE_QUERY = """
query ProofTree($sessionId: String!) {
  proofTree(sessionId: $sessionId) {
    nodes { name description status value source confidence tier strategy dependencies }
    summary
  }
}
"""

MESSAGES_QUERY = """
query Messages($sessionId: String!) {
  messages(sessionId: $sessionId) {
    messages { role content timestamp }
  }
}
"""

PROMPT_CONTEXT_QUERY = """
query PromptContext($sessionId: String!) {
  promptContext(sessionId: $sessionId) {
    systemPrompt tokenCount
  }
}
"""

DATA_SOURCES_QUERY = """
query DataSources($sessionId: String!) {
  dataSources(sessionId: $sessionId) {
    databases { name type uri description connected fromConfig source }
    apis { name type baseUrl description connected fromConfig source }
    documents { name type format url description source }
    emails { name type address provider }
    facts { name value source }
  }
}
"""

DATABASES_QUERY = """
query Databases($sessionId: String!) {
  databases(sessionId: $sessionId) {
    databases { name type uri description connected }
  }
}
"""

GLOSSARY_QUERY = """
query Glossary($sessionId: String!, $scope: String, $domain: String) {
  glossary(sessionId: $sessionId, scope: $scope, domain: $domain) {
    terms {
      name displayName definition domain domainPath
      parentId parentVerb aliases semanticType status
      glossaryStatus entityId tags ignored canonicalSource
    }
    total
  }
}
"""

GLOSSARY_TERM_QUERY = """
query GlossaryTerm($sessionId: String!, $name: String!) {
  glossaryTerm(sessionId: $sessionId, name: $name) {
    name displayName definition domain domainPath
    parentId parentVerb aliases semanticType status
    glossaryStatus entityId tags ignored canonicalSource
    connectedResources {
      entityName entityType
      sources { documentName source section url }
    }
    children { name displayName parentVerb }
    relationships { id subject verb object }
    clusterSiblings
    spanningDomains
  }
}
"""

OBJECTIVES_QUERY = """
query Objectives($sessionId: String!) {
  objectives(sessionId: $sessionId) { index text }
}
"""

ACTIVE_DOMAINS_QUERY = """
query ActiveDomains($sessionId: String!) {
  activeDomains(sessionId: $sessionId)
}
"""

# ---------------------------------------------------------------------------
# ConstatClient-level queries
# ---------------------------------------------------------------------------

DOMAINS_QUERY = """
query Domains { domains { domains { filename name description } } }
"""

SKILLS_QUERY = """
query Skills { skills { skills { name description } } }
"""

SKILL_QUERY = """
query Skill($name: String!) {
  skill(name: $name) { name description content }
}
"""

LEARNINGS_QUERY = """
query Learnings($category: String) {
  learnings(category: $category) { learnings { id text category createdAt } total }
}
"""

AGENTS_QUERY = """
query Agents($sessionId: String!) {
  agents(sessionId: $sessionId) { name description isActive }
}
"""

# ---------------------------------------------------------------------------
# Write mutations
# ---------------------------------------------------------------------------

ADD_FACT = """
mutation AddFact($sessionId: String!, $name: String!, $value: JSON!, $persist: Boolean) {
  addFact(sessionId: $sessionId, name: $name, value: $value, persist: $persist) {
    status fact { name value source isPersisted }
  }
}
"""

FORGET_FACT = """
mutation ForgetFact($sessionId: String!, $factId: Int!) {
  forgetFact(sessionId: $sessionId, factId: $factId) { status }
}
"""

CREATE_GLOSSARY_TERM = """
mutation CreateGlossaryTerm($sessionId: String!, $input: GlossaryTermInput!) {
  createGlossaryTerm(sessionId: $sessionId, input: $input) {
    name displayName definition
  }
}
"""

DELETE_GLOSSARY_TERM = """
mutation DeleteGlossaryTerm($sessionId: String!, $name: String!) {
  deleteGlossaryTerm(sessionId: $sessionId, name: $name)
}
"""

REFINE_DEFINITION = """
mutation RefineDefinition($sessionId: String!, $name: String!) {
  refineDefinition(sessionId: $sessionId, name: $name) {
    definition
  }
}
"""

GENERATE_GLOSSARY = """
mutation GenerateGlossary($sessionId: String!) {
  generateGlossary(sessionId: $sessionId) { status message }
}
"""

TOGGLE_TABLE_STAR = """
mutation ToggleTableStar($sessionId: String!, $tableName: String!) {
  toggleTableStar(sessionId: $sessionId, tableName: $tableName) { starred }
}
"""

DELETE_TABLE = """
mutation DeleteTable($sessionId: String!, $tableName: String!) {
  deleteTable(sessionId: $sessionId, tableName: $tableName)
}
"""

TOGGLE_ARTIFACT_STAR = """
mutation ToggleArtifactStar($sessionId: String!, $artifactId: Int!) {
  toggleArtifactStar(sessionId: $sessionId, artifactId: $artifactId) { starred }
}
"""

DELETE_ARTIFACT = """
mutation DeleteArtifact($sessionId: String!, $artifactId: Int!) {
  deleteArtifact(sessionId: $sessionId, artifactId: $artifactId)
}
"""

ADD_DATABASE = """
mutation AddDatabase($sessionId: String!, $input: DatabaseAddInput!) {
  addDatabase(sessionId: $sessionId, input: $input) { name type }
}
"""

REMOVE_DATABASE = """
mutation RemoveDatabase($sessionId: String!, $name: String!) {
  removeDatabase(sessionId: $sessionId, name: $name) { status }
}
"""

ADD_API = """
mutation AddApi($sessionId: String!, $input: ApiAddInput!) {
  addApi(sessionId: $sessionId, input: $input) { name type }
}
"""

REMOVE_API = """
mutation RemoveApi($sessionId: String!, $name: String!) {
  removeApi(sessionId: $sessionId, name: $name) { status }
}
"""

ADD_DOCUMENT_URI = """
mutation AddDocumentUri($sessionId: String!, $input: DocumentUriInput!) {
  addDocumentUri(sessionId: $sessionId, input: $input) { status }
}
"""

SET_ACTIVE_DOMAINS = """
mutation SetActiveDomains($sessionId: String!, $domains: [String!]!) {
  setActiveDomains(sessionId: $sessionId, domains: $domains)
}
"""

RESET_CONTEXT = """
mutation ResetContext($sessionId: String!) {
  resetContext(sessionId: $sessionId)
}
"""

FLAG_ANSWER = """
mutation FlagAnswer($sessionId: String!, $text: String!) {
  flagAnswer(sessionId: $sessionId, text: $text) { status }
}
"""

COMPACT_LEARNINGS = """
mutation CompactLearnings {
  compactLearnings { status message }
}
"""

CREATE_RULE = """
mutation CreateRule($text: String!, $category: String) {
  createRule(text: $text, category: $category) { id text category }
}
"""

UPDATE_RULE = """
mutation UpdateRule($ruleId: Int!, $text: String!) {
  updateRule(ruleId: $ruleId, text: $text) { id text category }
}
"""

DELETE_RULE = """
mutation DeleteRule($ruleId: Int!) {
  deleteRule(ruleId: $ruleId) { status }
}
"""
