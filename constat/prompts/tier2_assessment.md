Tier 1 resolution failed for: {fact_name}
Description: {fact_description}

Resolved premises in current plan:
{resolved_premises}

Pending premises in current plan:
{pending_premises}

Available data sources (already searched, fact not found directly):
{available_sources}

Assess the best resolution strategy:

STRATEGY: DERIVABLE | KNOWN | USER_REQUIRED

CRITICAL: DERIVABLE requires a plan with 2+ DISTINCT inputs being composed.
- Valid: "X = A / B" (two inputs composed with formula)
- Valid: "X = filter(A, condition from B)" (two inputs)
- INVALID: "try looking up synonym Y instead" (single lookup, REJECTED)
- INVALID: "search for alternative_name" (synonym hunting, REJECTED)

If you cannot devise a formula with 2+ distinct inputs, do NOT use DERIVABLE.

CONFIDENCE: 0.0-1.0
REASONING: <brief explanation of why this strategy>

If DERIVABLE:
  FORMULA: <computation formula, must reference 2+ inputs>
  INPUTS: <list of (input_name, source) tuples, e.g., [("salaries", "premise:P1"), ("industry_avg", "llm_knowledge")]>

If KNOWN:
  VALUE: <the answer - only use for general/industry knowledge you're confident about>
  CAVEAT: <any limitations or uncertainty>

If USER_REQUIRED:
  QUESTION: <clear question to ask the user>

Respond in valid JSON format:
{{
  "strategy": "DERIVABLE" | "KNOWN" | "USER_REQUIRED",
  "confidence": 0.0-1.0,
  "reasoning": "...",
  "formula": "..." or null,
  "inputs": [["name", "source"], ...] or null,
  "value": ... or null,
  "caveat": "..." or null,
  "question": "..." or null
}}