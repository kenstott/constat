Analyze this question for ambiguity. Determine if critical parameters are missing that would significantly change the analysis.

Question: "{problem}"

Available data sources (databases AND APIs - both are valid data sources):
{schema_overview}{api_overview}{doc_overview}{user_facts}{learnings_text}{session_tables}

DATA AVAILABILITY IS NOT YOUR CONCERN. The system has databases, APIs, and documents listed above.
Assume they contain whatever the user is asking about. Schema resolution, table discovery, and
column matching happen automatically during execution. You will NEVER ask about any of these.

If an API/database/document is listed above and the question references related data, the question is CLEAR.
If a user fact provides needed information (like user_email), USE IT - do not ask again.
If session tables exist and the question references a dataset, match it to the most relevant session table.
If the user references a document by name and it appears in Reference Documents, the question is CLEAR.

ONLY ask about SCOPE and APPROACH - things that affect how to structure the analysis:
1. Geographic scope (country, region, state, etc.) - unless an API provides this
2. Time period (date range, year, quarter, etc.)
3. Quantity limits (top N, threshold values)
4. Category/segment filters (which products, customer types, etc.)
5. Comparison basis (compared to what baseline?)

FORBIDDEN QUESTION TOPICS — asking about ANY of these means you failed:
- Whether a database/table/column contains specific data
- Whether data "exists" or is "available"
- Which table, column, or field to use
- Whether to use a database vs document vs API
- Whether to extract from a document or query a database
- Schema structure, field names, data types
- doc_read(), schema exploration, database discovery
These are implementation details. The user cannot answer them and should not be asked.

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
If the user's question specifies what data they want and what analysis to perform, it is CLEAR.
The user's question above specifies exactly what to do — if you ask about data availability, you have failed.
