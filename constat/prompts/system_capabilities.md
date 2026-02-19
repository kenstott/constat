# Constat System Capabilities

## Overview

Constat is a multistep reasoning engine for data analysis. Its AI assistant is named **Vera** (Latin for "truth"). Vera breaks complex questions into executable analytical steps, queries real data sources, and shows transparent reasoning. Unlike chat-based AI, every conclusion is grounded in actual data with visible steps.

## Multi-Step Reasoning

When you ask a question, Constat:
1. **Plans** — Decomposes your question into a series of analytical steps
2. **Executes** — Runs each step: database queries, API calls, computations, or LLM reasoning
3. **Shows work** — Intermediate results are stored as inspectable tables
4. **Synthesizes** — Combines findings into a direct answer
5. **Suggests** — Recommends follow-up analyses

Steps form a **directed acyclic graph (DAG)** — independent steps run in parallel automatically. Each step can gather facts from databases, APIs, documents, user input, LLM knowledge, or derivation from prior steps.

## Proof Mode (Auditable Reasoning)

For formal verification of claims:
- Activated with the `/prove` command or by asking questions with "verify", "prove", "validate"
- Your question becomes a **hypothesis** to prove or disprove
- Works backwards: **recursively decomposes claims** until grounded in verifiable facts
- Each inference must produce evidence supporting or refuting the claim
- Results include **confidence levels** and caveats
- Produces a full **audit trail** for compliance and review
- Domain rules and constraints are strictly enforced throughout

## Glossary & Taxonomy

Constat automatically builds a **business glossary** from your data:
- Discovers entities, concepts, and terms from database schemas and queries
- Generates **definitions** grounded in actual data structures
- Organizes terms into a **taxonomy** (parent/child hierarchy)
- Identifies **aliases** (e.g., "Purchase Order" = "Sales Order" = "Customer Order")
- Terms go through an **approval workflow**: draft → defined → approved
- **Domain promotion** allows promoting terms to be shared across the organization
- **Entity grounding** links glossary terms to database columns and tables
- Search queries are resolved against glossary terms for consistent interpretation

## Relationship Extraction

Constat discovers and tracks relationships between entities:
- **SVO relationships** — Subject-Verb-Object (e.g., "order contains order item")
- **FK relationships** — Foreign key relationships from database schemas
- **Cross-cutting inference** — Relationships inferred across data sources
- Relationships are visualized as a **graph**
- Users can add, edit, or remove relationships manually

## Clarification & Ambiguity

When a question is ambiguous, Constat asks **clarifying questions** before proceeding:
- Detects missing parameters (time period, geographic scope, thresholds, categories)
- Presents focused questions with suggested answers
- In proof mode, personal value questions are deferred to lazy fact resolution
- Users can skip clarification to proceed with defaults
- Clarification can be disabled in session configuration

## Assumptions & Uncertainty

Constat tracks facts and assumptions explicitly:
- **User facts** — Values provided by the user (e.g., "my role is CFO")
- **Derived facts** — Values computed during analysis
- **Unresolved facts** — Information needed but not yet provided; the system asks for these during execution
- Users can **correct** facts at any time, triggering re-analysis
- The system admits uncertainty rather than guessing
- All assumptions are visible and challengeable

## Skills

Reusable domain knowledge modules:
- Encapsulate analytical patterns (e.g., "attrition risk analysis", "revenue forecasting")
- Can be **created from proof results** — a proven analysis becomes a reusable skill
- Skills plug into the planning and execution pipeline
- Can be shared, downloaded, activated, or deactivated
- Managed with `/skill`, `/skills`, `/skill-create`, `/skill-draft`, `/skill-download`

## Roles

Domain-specific personas that shape how Vera responds:
- Each role has expertise, focus areas, and communication style
- **Dynamic role matching** — the system can auto-select the best role for a question
- Custom roles can be created, edited, or deleted
- Roles influence planning, code generation, and answer synthesis
- Managed with `/role`, `/roles`, `/role-create`, `/role-draft`

## Learnings & Corrections

Constat learns from user corrections:
- **User corrections** — "Actually, use revenue not gross sales" becomes a persistent rule
- **NL corrections** — Natural language hints are detected and stored
- **Rules** — Explicit business rules that guide future analyses
- Learnings are applied during ambiguity detection, planning, and execution
- Managed with `/learnings`, `/correct`, `/rule`, `/rule-edit`, `/rule-delete`

## Data Sources

Constat connects to multiple data source types:
- **Databases** — PostgreSQL, MySQL, SQLite, DuckDB, and more
- **Structured files** — CSV, Excel, Parquet, JSON (loaded as queryable tables)
- **Documents** — PDF, Word, PowerPoint for reference and context
- **APIs** — REST endpoints for live data
- **Progressive discovery** — metadata is fetched on-demand, not preloaded, allowing connection to large data estates

## Artifacts

Constat produces rich output:
- **Tables** — Intermediate and final result tables, inspectable and queryable
- **Visualizations** — Charts and graphs generated during analysis
- **Reports** — Synthesized narrative answers with data backing
- **Exports** — Results can be exported or downloaded
- Managed with `/artifacts`, `/export`, `/download-code`

## Commands

Key commands available:
- `/prove` — Re-run analysis in auditable proof mode
- `/tables` — List all session tables
- `/show <table>` — Display a specific table
- `/query <sql>` — Run SQL against session data
- `/facts` — Show all known facts
- `/correct` — Correct a previous assumption or result
- `/role` / `/roles` — View or switch roles
- `/skill` / `/skills` — View or manage skills
- `/learnings` — View learned rules and corrections
- `/discover` — Discover available data sources
- `/databases` / `/apis` / `/documents` — List configured data sources
- `/export` — Export results
- `/reset` — Reset the current session
- `/help` — Show all available commands
