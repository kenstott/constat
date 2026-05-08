# Skills Architecture

**Status:** Design

## Overview

Skills are domain-specific knowledge modules (`SKILL.md` files) that inject context into the LLM during planning. They follow the [Agent Skills open standard](https://agentskills.io).

This document covers two concerns:
1. The existing skill creation flow (manual + AI-drafted)
2. A new "Convert Proof to Skill" option that captures verified derivation knowledge as a reusable skill

## Current Skill Creation

### Storage

```
.constat/{user_id}/skills/{skill-name}/
├── SKILL.md          # Required — YAML frontmatter + markdown body
├── scripts/          # Optional executable code
├── references/       # Optional documentation
└── assets/           # Optional templates, icons, etc.
```

### SKILL.md Format

```yaml
---
name: skill-name
description: What this skill does
allowed-tools: [list_tables, get_table_schema, run_sql]
---

Markdown body with:
- Key metrics and calculations
- SQL query patterns
- Domain terminology
- Best practices
```

### Creation Paths (Current)

| Path | Trigger | Input | Output |
|------|---------|-------|--------|
| Manual | Plus button in Skills panel | Name, description, tools, body | SKILL.md written directly |
| AI Draft | "Draft with AI" button | Name + description | LLM generates full SKILL.md |

Both paths live in the ArtifactPanel Skills accordion section.

## Decision: Proof-to-Skill Conversion

### Problem

A completed proof contains verified facts, derivation strategies, source bindings, and confidence scores — exactly the kind of domain knowledge that makes a good skill. Today this knowledge is discarded after viewing. Users who want to capture it must manually author a skill from memory.

### Proposal

When a completed proof is in scope, add a **"Save as Skill"** action to the ProofDAGPanel. This is a third creation path alongside manual and AI-drafted:

| Path | Trigger | Input | Output |
|------|---------|-------|--------|
| Manual | Plus button in Skills panel | Name, description, tools, body | SKILL.md written directly |
| AI Draft | "Draft with AI" button | Name + description | LLM generates full SKILL.md |
| **From Proof** | **"Save as Skill" in ProofDAGPanel** | **Proof nodes + summary + optional name** | **LLM distills proof into SKILL.md** |

### Constraints

- At most **one proof** is in scope at a time (per `proofStore`)
- The proof must be **completed** (`hasCompletedProof === true`) before conversion
- The proof summary (LLM-generated) should be available but is not strictly required

### UX Flow

```
ProofDAGPanel (proof completed)
  │
  ├─ Existing: "View Summary" tab shows LLM-generated proof summary
  │
  └─ New: "Save as Skill" button (appears after proof_complete)
       │
       ├─ Click → inline form (name input, optional description override)
       │          Pre-populated description from proof summary first line
       │
       └─ Confirm → POST /api/skills/from-proof
                     │
                     ├─ LLM distills proof nodes into skill content
                     ├─ Creates SKILL.md with verified patterns
                     └─ Skill appears in ArtifactPanel Skills list
```

The button only appears when `hasCompletedProof` is true. It sits alongside the existing summary view — not replacing any current UI.

### What Gets Captured

The skill gets two artifacts: a **SKILL.md** (LLM-distilled patterns) and a **scripts/proof.py** (executable inference code).

#### SKILL.md — Reusable Patterns

| Proof Data | Skill Output |
|------------|-------------|
| Resolved fact names + values | Metric definitions table |
| Fact sources (DATABASE, API, DOCUMENT) | Source reference section |
| Derivation strategies (SQL, aggregation, join) | SQL patterns as code blocks |
| Fact dependencies (DAG edges) | Derivation order / methodology notes |
| Confidence scores | Reliability annotations |
| Proof summary | Skill description + overview |

#### scripts/proof.py — Executable Inference Code

The existing `GET /{session_id}/download-inference-code` endpoint (`constat/server/routes/data.py:2487`) already generates a self-contained Python script containing:

- `_DataStore` class (in-memory DuckDB + DataFrame registry)
- API helpers (GraphQL/REST) with configured URLs
- Database helpers (SQLAlchemy engines)
- `llm_map()` stub for fuzzy mapping
- Each inference as a named function with its resolved code
- `run_proof(**premises)` entry point with constant premises as kwargs
- `__main__` block for standalone execution

This generation logic is reused for the skill's `scripts/` directory. The script is **not regenerated** — it's the same code the user would get from the download button, but persisted inside the skill.

```
.constat/{user_id}/skills/quarterly-retention/
├── SKILL.md              # LLM-distilled patterns (new)
└── scripts/
    └── proof.py          # Inference code (reused from download-inference-code)
```

The SKILL.md body references the script:

```markdown
## Executable Proof

A verified proof script is available at `scripts/proof.py`.
Run standalone: `python scripts/proof.py`

Premise parameters (configurable via kwargs to `run_proof()`):
- `fiscal_year_start="April 1"`
- `churn_threshold=0.05`
```

### Decision: Extract Script Generation

The script generation logic currently lives inline in the `download_inference_code` route handler (~230 lines). Extract it into a reusable function:

```python
# In constat/server/routes/data.py or a new constat/core/inference_export.py

def generate_inference_script(
    inferences: list[dict],
    premises: list[dict],
    apis: list[dict],
    databases: list[dict],
    session_id: str,
) -> str:
    """Generate a self-contained Python script from inference codes.

    Returns the script content as a string. Used by both:
    - GET /download-inference-code (returns as HTTP response)
    - POST /skills/from-proof (writes to skill scripts/ dir)
    """
```

The existing `download_inference_code` endpoint becomes a thin wrapper:

```python
@router.get("/{session_id}/download-inference-code")
async def download_inference_code(...):
    # ... gather inferences, premises, apis, databases (unchanged)
    script_content = generate_inference_script(inferences, premises, apis, databases, session_id)
    return Response(content=script_content, media_type="text/x-python", ...)
```

### Backend

#### New Endpoint

```
POST /api/skills/from-proof?session_id={id}

Request:
{
  "name": "quarterly-retention-analysis",    # required
  "description": "override description"      # optional
}

Response:
{
  "name": "quarterly-retention-analysis",
  "content": "---\nname: quarterly-retention-analysis\n...",
  "description": "Verified retention metric derivations...",
  "has_script": true
}
```

#### Implementation

```python
def skill_from_proof(
    name: str,
    proof_nodes: list[dict],
    proof_summary: str | None,
    original_problem: str,
    llm,
    description: str | None = None,
) -> tuple[str, str]:
    """Distill a completed proof into a SKILL.md.

    Returns (content, description).
    """
```

The endpoint:

1. Pulls proof state from session (`last_proof_result`)
2. Calls `skill_from_proof()` to generate SKILL.md via LLM
3. Calls `generate_inference_script()` to get the proof script
4. Creates the skill directory, writes SKILL.md, writes `scripts/proof.py`

#### Prompt Strategy

The key distinction: a proof captures **specific verified values** ("Q4 churn was 12.3%"). A skill should capture **reusable patterns** ("To calculate churn rate, use `status = 'C'` in `sales_db.customers`, aggregate by period"). The executable script preserves the specific derivation for reproducibility, while the SKILL.md generalizes.

```
You are converting a verified proof into a reusable skill.

The proof verified specific claims. Your job is to extract the
REUSABLE PATTERNS — not the specific values.

The exact inference code is saved separately as scripts/proof.py.
Your SKILL.md should focus on the domain knowledge:

Focus on:
- Which tables/APIs/documents were used and how
- SQL patterns that resolved facts successfully
- Join paths and aggregation strategies
- Source-specific field mappings and enum values
- Derivation order when facts depend on other facts
- Reference scripts/proof.py for the executable derivation

Do NOT include:
- Specific numeric results from this proof run
- Date ranges or filters specific to the original query
- One-time observations that won't generalize
- Raw code (that's in scripts/proof.py)
```

### Frontend

#### ProofDAGPanel Changes

Add to the panel footer (next to existing close/summary controls):

```tsx
// Only when proof is complete
{hasCompletedProof && (
  <button onClick={openSkillFromProofForm}>
    Save as Skill
  </button>
)}
```

Clicking opens a minimal inline form (skill name + confirm). On confirm:

1. `POST /api/skills/from-proof` (backend handles both SKILL.md and scripts/)
2. On success, refresh skills list in `useArtifactStore`
3. Show toast confirmation

#### No Changes to ArtifactPanel

The existing Skills accordion is unaffected. The new skill appears in the list like any other skill — it's just created from a different entry point. The `scripts/proof.py` is visible when expanding the skill directory.

### API Route

```python
# In constat/server/routes/skills.py

@router.post("/from-proof")
async def create_skill_from_proof(
    request: SkillFromProofRequest,
    session_id: str = Query(...),
):
    """Create a skill from a completed proof's derivation trace."""
    session = session_manager.get_session(session_id)
    proof_result = session.last_proof_result

    # 1. LLM-distilled SKILL.md
    content, description = skill_from_proof(
        name=request.name,
        proof_nodes=proof_result["proof_nodes"],
        proof_summary=proof_result.get("summary"),
        original_problem=proof_result.get("problem", ""),
        llm=session.router,
        description=request.description,
    )

    # 2. Executable inference script (reuse existing generation)
    inferences = session.history.list_inference_codes(session.session_id)
    premises = session.history.list_inference_premises(session.session_id)
    apis, databases = _gather_source_configs(session)
    script = generate_inference_script(inferences, premises, apis, databases, session_id)

    # 3. Write skill directory
    skill_manager.create_skill(name=request.name, prompt="placeholder", description=description)
    skill_manager.update_skill_content(request.name, content)
    scripts_dir = skill_manager.get_skill_dir(request.name) / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "proof.py").write_text(script)

    return {
        "name": request.name,
        "content": content,
        "description": description,
        "has_script": True,
    }
```

## Implementation

### Phase 1: Extract Script Generation

1. Extract `generate_inference_script()` from `download_inference_code` route
2. Place in `constat/core/inference_export.py` (or inline in `data.py`)
3. Refactor `download_inference_code` to call the extracted function
4. No behavior change — pure refactor

### Phase 2: Backend Skill-from-Proof

1. Cache `prove_conversation()` result on session as `last_proof_result`
2. Add `skill_from_proof()` to `constat/core/skills.py`
3. Add `POST /api/skills/from-proof` route
4. Route writes both SKILL.md and `scripts/proof.py`

### Phase 3: Frontend

1. Add "Save as Skill" button to `ProofDAGPanel` (gated on `hasCompletedProof`)
2. Add inline name input form
3. Wire to new API endpoint
4. Refresh skills list on success

### Critical Files

| File | Changes |
|------|---------|
| `constat/server/routes/data.py` | Extract `generate_inference_script()` from `download_inference_code` |
| `constat/session.py` | Cache `last_proof_result` after `prove_conversation()` |
| `constat/core/skills.py` | Add `skill_from_proof()`, add `get_skill_dir()` |
| `constat/server/routes/skills.py` | Add `/from-proof` endpoint |
| `constat-ui/src/components/proof/ProofDAGPanel.tsx` | "Save as Skill" button + form |
| `constat-ui/src/api/skills.ts` | Add `createSkillFromProof()` client method |