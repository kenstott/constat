# Constat Value Proposition vs Claude + MCP

## Context

Claude Desktop + MCP provides stateless tool calling with a growing ecosystem of connectors. This document captures what constat provides beyond that baseline.

## Core Differentiators

### 1. Domain DAG — Composable, Tiered Configuration

Domains bundle sources, glossaries, agents, skills, tools, and learned rules as a single composable unit. Five-tier inheritance (Global → Organization → Domain → Session → User) with override semantics. Multi-domain sessions combine configurations with conflict resolution.

Claude + MCP: flat list of servers, no composability, no inheritance, no bundling.

### 2. Indexed Domain Knowledge — Entity Extraction, Linking, and Grounded Reasoning

A multi-stage pipeline that builds a queryable knowledge graph from raw sources:

1. **Vectorization** — All ingested content embedded (HNSW + BM25 + RRF reranking) for semantic search across documents, schemas, and API specs.
2. **NER extraction** — Automatic entity extraction from every source at ingest. No manual tagging.
3. **Entity → chunk linking** — Every entity traces back to the source chunk that produced it. Provenance is built-in, not retrofitted.
4. **Relationship inference** — Entities connected via co-occurrence, explicit references, and schema foreign keys. The system knows "Q4 revenue" involves `orders.total`, `products.category`, and date filtering on `orders.order_date`.
5. **Clustering** — Related entities grouped automatically (e.g., "revenue", "ARR", "MRR" cluster together). Spanning entities identified across multiple sources.
6. **Glossary grounding** — Persistent, reusable business definitions with aliases, categories, and grounding status (connected to source data vs. user-defined).
7. **Client-side cache** — Entity/glossary data cached locally, deduped, NER-fingerprinted. The planner has grounded entity context *before* making any API calls — zero-latency lookups during query planning.

The result: the planner selects the right tables/columns/APIs based on indexed domain knowledge, not guesswork.

Claude + MCP: reasons over tool descriptions only. No entity awareness, no relationship graph, no semantic understanding of actual data content. Discovery is trial and error.

PromptQL: manually curated "context graph" (wiki-style). Depends on users writing and maintaining entries. No automated extraction, no source grounding, no caching.

### 3. Proof DAG — Cumulative Reasoning Graph

Multi-step questions produce a DAG of fact dependencies. Independent facts resolve in parallel waves. Cached intermediates are reused across steps. "Now break that down by region" extends the graph — it doesn't re-derive everything. Source refresh triggers invalidation by walking the graph.

Claude + MCP: each question is independent. No graph, no caching, no invalidation, no parallel scheduling.

### 4. Integrated Session Memory — Typed, Queryable, Persistent Artifacts

Per-step artifacts are typed (DataFrame, chart, dict, string), versioned, and registered as DuckDB views. Downstream steps can SQL query prior artifacts. Session restore brings all artifacts back into the same queryable namespace. Artifacts are wired into the proof DAG — upstream staleness propagates.

MCP memory servers (Mem0, etc.) provide generic key-value recall. Constat's memory is load-bearing infrastructure, not bolted-on storage.

### 5. Learning Engine — Automated Capture → Rules → Fine-Tuning

System automatically observes query patterns, failures, corrections, and successful strategies. Users can also teach rules explicitly. Raw observations are compacted into rules when thresholds are met. Two output paths:

- **Prompt injection** — Rules inserted into planner context at query time. Immediate effect.
- **Fine-tune export** — Exemplar generator exports rules as training data (OpenAI/Alpaca JSONL). The system generates its own fine-tuning dataset.

Rules are scoped (database type, instance, domain, global) and travel with the domain DAG.

Claude + MCP: starts blank every session. No accumulation, no self-improvement, no fine-tuning pipeline.

### 6. Structured Provenance — Auditable Derivation Chains

Each fact records its dependencies, SQL queries, code, row counts, and source references as machine-readable derivation chains. Required for compliance/fintech use cases where "Claude explained its reasoning" is insufficient.

Claude + MCP: can explain reasoning in natural language. Cannot produce structured, traversable provenance.

### 7. Regression Testing — Self-Validation

Golden question harness with assertion types (term, grounding, relationship, end-to-end). Two-phase testing: metadata checks then LLM judge. Bug queue tracks known failures. Domain-specific test suites travel with the domain DAG.

Claude + MCP: no test harness concept. Cannot validate its own domain coverage.

### 8. Multi-Surface Access — Web, REPL, Jupyter

Three interfaces sharing the same session state:

- **Web UI** — Full-featured browser interface with artifact panel, glossary, proof visualization.
- **REPL/CLI** — Interactive terminal with full session state, follow-ups, data source attachment, artifact inspection. Works over SSH, in CI, headless. Enables scripting and automation.
- **Jupyter integration** — Notebooks use constat as a library, mixing LLM-driven analysis with manual code in the same session context. Data scientists stay in their native workflow.

All three share session state, artifacts, and proof graphs. Start in Jupyter, resume in the web UI, script regression tests from the CLI.

Claude Desktop: chat interface only. PromptQL: web-only workspace. Neither offers CLI, notebook integration, or automation hooks.

### 9. Scale — Data Beyond Context Windows

DuckDB federation (ATTACH, views, Arrow zero-copy, `read_*_auto`) puts millions of rows into a queryable namespace. Precise SQL semantics for aggregations, window functions, exact numeric results.

Claude + MCP: in-context joins work for small result sets. Fails at scale or when precision/reproducibility is required.

## What MCP *Does* Cover

- **Basic tool calling** — MCP tools can query databases, read files, call APIs
- **Schema exposure** — SQL MCP servers expose table metadata
- **Session memory** — MCP memory servers (Mem0, etc.) provide key-value recall
- **Small-data joins** — Claude can reason over JSON responses from multiple tools for small result sets

These are necessary but not sufficient for analytical reasoning at scale.

## Summary

Claude + MCP is a stateless tool-calling layer. Constat is a **stateful analytical reasoning engine** that provides composable domain configuration, indexed domain knowledge, cumulative reasoning graphs, integrated persistent memory, automated learning, structured provenance, self-validation, and multi-surface access (web, REPL, Jupyter). MCP is one of constat's I/O channels, not a replacement for its orchestration layer.
