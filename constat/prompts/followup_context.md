Previous work in this session:

{scratchpad_context}

**EXISTING TABLES** (use these exact names when updating):
{existing_tables_list}

Available state variables:
{existing_state}

**CRITICAL — DATASET REUSE RULES**:
- When modifying or extending existing analysis: use the EXACT table name from above (e.g., step produces `{first_table_name}` not a new variant)
- OVERWRITE existing datasets when the modification replaces or extends previous results
- Only create a NEW name when adding a genuinely separate dataset alongside existing ones
- Reference and build on existing tables/state rather than recreating from scratch
- The final result should only contain items that address the complete plan (original + extensions)

Follow-up question: {question}
