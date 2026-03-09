Analyze this question for ambiguity. Determine if critical parameters are missing that would significantly change the analysis.

Question: "{problem}"

Available data sources (databases AND APIs - both are valid data sources):
{schema_overview}{api_overview}{doc_overview}{user_facts}{learnings_text}{session_tables}

IMPORTANT: If an API can provide the data needed for the question, the question is CLEAR.
For example, if the question asks about countries and a countries API is available, that's sufficient.
If a user fact provides needed information (like user_email for sending results), USE IT - do not ask again.
If session tables exist and the question references a dataset, match it to the most relevant session table - do NOT ask which dataset.
If the user references a specific document by name AND that document appears in the Reference Documents list, the question is CLEAR — the document is accessible via doc_read(). NEVER ask the user to "confirm access" to a listed document.

ONLY ask about SCOPE and APPROACH - things that affect how to structure the analysis:
1. Geographic scope (country, region, state, etc.) - unless an API provides this
2. Time period (date range, year, quarter, etc.)
3. Quantity limits (top N, threshold values)
4. Category/segment filters (which products, customer types, etc.)
5. Comparison basis (compared to what baseline?)

NEVER ask about data sources, data availability, schema contents, or implementation strategy. Specifically:
- Do NOT ask which table, column, or field contains the data
- Do NOT ask whether to use a database vs a document vs an API
- Do NOT ask whether to extract data from a document or query a database
- Do NOT suggest doc_read(), schema exploration, or database queries as options
- Do NOT ask which source is "primary" — source resolution is automatic
These are implementation details handled by the system, not user decisions.

{personal_values_guidance}

If the question is CLEAR ENOUGH to proceed (even with reasonable defaults), respond:
CLEAR

If critical parameters are missing that would significantly change results, respond:
AMBIGUOUS
REASON: <brief explanation of what's unclear>
QUESTIONS:
Q1: <specific clarifying question ending with ?>
SUGGESTIONS: <suggestion1> | <suggestion2> | <suggestion3>
WIDGET: <choice|curation|ranking|mapping|table>
Q2: <specific clarifying question ending with ?>
SUGGESTIONS: <suggestion1> | <suggestion2>
(max 3 questions, each question MUST end with ?)

WIDGET selection rules (pick the best fit for each question):
- **choice** (default — omit WIDGET line): 2-4 mutually exclusive options
- **curation**: subset selection from 5+ items — list ALL relevant items in SUGGESTIONS (no limit)
- **ranking**: ordering/prioritizing 3+ items — list all in SUGGESTIONS
- **mapping**: pairing between two lists — use SUGGESTIONS_LEFT: Label: item1 | item2 and SUGGESTIONS_RIGHT: Label: item1 | item2 instead of SUGGESTIONS
- **table**: multi-parameter input — use COLUMNS: col1 | col2 and COLUMN_TYPES: text | boolean | select instead of SUGGESTIONS

Only flag as AMBIGUOUS if the missing info would SIGNIFICANTLY change the analysis approach.
Do NOT flag as ambiguous if an available API can fulfill the data requirement.
Do NOT ask about information already provided in Known User Facts.
Do NOT ask about topics already resolved by Learned Rules - treat those as settled decisions.

CRITICAL: Only suggest options that the USER can answer — scope, approach, business intent.
- NEVER suggest schema exploration, doc_read(), field inspection, or database discovery as options — those happen automatically during execution.
- NEVER ask which table, column, or field contains the data — the system resolves this.
- If the user asks about data types not in the schema, that is NOT an ambiguity — the system will handle it or fail gracefully.

DOCUMENT-AWARE: When Reference Documents are listed and relevant to the question, treat the question as CLEAR — the system will use the document automatically. Do NOT ask whether to use the document.
