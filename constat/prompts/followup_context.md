Previous work in this session:

{scratchpad_context}

**EXISTING TABLES** (use these exact names when updating):
{existing_tables_list}

Available state variables:
{existing_state}

**CRITICAL — DATASET REUSE RULES**:
- **NEVER re-derive data that already exists in an EXISTING TABLE above.** If the data is there, plan steps that LOAD it from the store — do NOT re-query the database for it. The existing tables are the source of truth for all prior work.
- When modifying or extending existing analysis: use the EXACT table name from above (e.g., step produces `{first_table_name}` not a new variant)
- OVERWRITE existing datasets when the modification replaces or extends previous results
- Only create a NEW name when adding a genuinely separate dataset alongside existing ones
- Reference and build on existing tables/state rather than recreating from scratch
- The final result should only contain items that address the complete plan (original + extensions)
- **ENHANCE = UPDATE SOURCE**: "enhance X", "add column to X", "enrich X" → the plan MUST end with a step that saves the updated X. Intermediate mapping/lookup tables are not the deliverable.
- **WRONG**: Step says "Query employees and performance reviews from database" when `most_recent_reviews` already exists above.
- **RIGHT**: Step says "Load most_recent_reviews from store and enrich with raise rate data"

Follow-up question: {question}
