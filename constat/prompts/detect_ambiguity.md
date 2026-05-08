Analyze this question for ambiguity. Determine if critical parameters are missing that would significantly change the analysis.

Question: "{problem}"

Available data sources (databases AND APIs - both are valid data sources):
{schema_overview}{api_overview}{doc_overview}{user_facts}{learnings_text}{session_tables}

IMPORTANT: If an API can provide the data needed for the question, the question is CLEAR.
For example, if the question asks about countries and a countries API is available, that's sufficient.
If a user fact provides needed information (like user_email for sending results), USE IT - do not ask again.
If session tables exist and the question references a dataset, match it to the most relevant session table - do NOT ask which dataset.

ONLY ask about SCOPE and APPROACH - things that affect how to structure the analysis:
1. Geographic scope (country, region, state, etc.) - unless an API provides this
2. Time period (date range, year, quarter, etc.)
3. Quantity limits (top N, threshold values)
4. Category/segment filters (which products, customer types, etc.)
5. Comparison basis (compared to what baseline?)

{personal_values_guidance}

If the question is CLEAR ENOUGH to proceed (even with reasonable defaults), respond:
CLEAR

If critical parameters are missing that would significantly change results, respond:
AMBIGUOUS
REASON: <brief explanation of what's unclear>
QUESTIONS:
Q1: <specific clarifying question ending with ?>
SUGGESTIONS: <suggestion1> | <suggestion2> | <suggestion3>
Q2: <specific clarifying question ending with ?>
SUGGESTIONS: <suggestion1> | <suggestion2>
(max 3 questions, 2-4 suggestions per question, each question MUST end with ?)

Only flag as AMBIGUOUS if the missing info would SIGNIFICANTLY change the analysis approach.
Do NOT flag as ambiguous if an available API can fulfill the data requirement.
Do NOT ask about information already provided in Known User Facts.
Do NOT ask about topics already resolved by Learned Rules - treat those as settled decisions.

CRITICAL: Only suggest options that can be answered with the AVAILABLE DATA shown above.
- Review the schema before suggesting options - don't suggest data that doesn't exist
- If the user asks about data types not in the schema, clarify what IS available instead
- Base suggestions on actual tables/columns shown above, not hypothetical data
- Provide practical suggested answers grounded in the actual available data

DOCUMENT-AWARE SUGGESTIONS:
- When Reference Documents are available AND relevant to the question, suggest using them
- If a document contains policies/guidelines that could answer a "what criteria" question,
  include a suggestion like "Based on [document name]" or "Use guidelines from [document]"
- Example: If user asks "how should salary increases be calculated" and a business_rules
  document exists with policies, suggest "Based on performance review guidelines in business_rules"
- This helps users leverage their internal documents instead of guessing criteria