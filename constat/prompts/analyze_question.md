Analyze this user question in one pass:

Question: "{problem}"
{source_context}
{fact_context}
{followup_context}

Perform these analyses:

1. FACT EXTRACTION: Extract any facts embedded in the question:
   - User context/persona (e.g., "my role as CFO" -> user_role: CFO)
   - Numeric values (e.g., "threshold of $50,000" -> revenue_threshold: 50000)
   - Preferences/constraints (e.g., "for the US region" -> target_region: US)
   - Time periods (e.g., "last quarter" -> time_period: last_quarter)

2. QUESTION CLASSIFICATION: Classify the question type:
   - META_QUESTION: About system capabilities ("what can you do?", "what data is available?")
   - DATA_ANALYSIS: Requires queries to configured data sources (databases, APIs) or computation
   - GENERAL_KNOWLEDGE: Can be answered from general LLM knowledge AND no configured data source has this data

   IMPORTANT: Prefer DATA_ANALYSIS if ANY configured source might have relevant data.

3. INTENT DETECTION: Identify ALL user intents in LOGICAL EXECUTION ORDER.
   A message can have MULTIPLE intents (e.g., "change threshold and redo" = MODIFY_FACT, then REDO).

   IMPORTANT: Order intents by when they should EXECUTE, not by word order in the text:
   - Temporal words override textual order: "redo AFTER changing threshold" -> MODIFY_FACT, REDO
   - "BEFORE", "FIRST", "THEN", "AFTER" indicate sequence
   - Priority words like "ALWAYS", "NEVER" push intents to the front
   - Dependencies matter: if B requires A's result, A comes first

   Possible intents:
   - REDO: Re-run analysis. Triggered by re-execution language ANYWHERE in message:
     "redo", "again", "retry", "rerun", "try again", "this time", "instead", etc.
     Also, implicit when user requests changes to a previous analysis.
   - MODIFY_FACT: Change a LITERAL VALUE ("change age to 50", "use $100k", "set to 10")
   - STEER_PLAN: Change METHODOLOGY/COMPUTATION ("use average of last 2", "compute X differently",
     "skip step", "different approach", "don't use that table", "change how X is calculated")
   - DRILL_DOWN: Explain ("why?", "show details", "break down")
   - REFINE_SCOPE: Filter ("only California", "exclude X", "just Q4")
   - CHALLENGE: Verify ("are you sure?", "double check", "confirm")
   - EXPORT: Save ("export to CSV", "download", "save")
   - EXTEND: Continue ("what about X?", "also check...")
   - PROVENANCE: Show proof ("where did that come from?", "audit trail")
   - CREATE_ARTIFACT: Create output ("create dashboard", "make chart", "generate report")
   - TRIGGER_ACTION: Execute action ("send email", "notify team")
   - COMPARE: Compare ("vs", "difference between", "compare to")
   - PREDICT: Forecast ("what if", "predict", "forecast")
   - LOOKUP: Simple lookup ("status of", "who owns", "when did")
   - ALERT: Set monitoring ("alert me when", "notify if")
   - SUMMARIZE: Condense results ("summarize", "give me the gist", "bottom line")
   - QUERY: Direct SQL query ("SELECT", "query the table", "run SQL")
   - RESET: Clear session ("start over", "clear everything", "fresh start")
   - NEW_QUESTION: A new query (default if nothing else applies)

4. CACHED FACT MATCH: If the question asks about a known fact, provide the answer.

5. EXECUTION MODE: Select the best mode for this request:
   - EXPLORATORY: Data analysis and creation - user wants to CREATE, ANALYZE, BUILD, or COMPUTE
     Examples: "Create an analysis...", "Show sales by region", "Build a dashboard"
   - PROOF: Verification with provenance - user needs PROOF, DEFENSIBLE conclusions, or AUDIT TRAIL
     Examples: "Prove that X", "Verify compliance", "Run in audit mode", "With full provenance"

   CRITICAL PRIORITY:
   1. EXPLICIT MODE REQUEST: If user says "audit mode", "auditable", "proof", "with provenance" -> PROOF
   2. EXPLICIT MODE REQUEST: If user says "exploratory mode" -> EXPLORATORY
   3. Otherwise, infer from task type

Respond in this exact format:
---
FACTS:
(list each as FACT_NAME: VALUE | brief description, or NONE if no facts)
---
QUESTION_TYPE: META_QUESTION | DATA_ANALYSIS | GENERAL_KNOWLEDGE
---
INTENTS:
(list IN ORDER as INTENT_NAME | optional extracted value, e.g., "MODIFY_FACT | threshold=$50k")
---
FACT_MODIFICATIONS:
(list as FACT_NAME: NEW_VALUE if user wants to change a fact, or NONE)
---
SCOPE_REFINEMENTS:
(list scope filters like "California", "Q4", "active users only", or NONE)
---
WANTS_BRIEF: YES or NO
(YES if user wants brief/concise output: "just show me", "quick answer", "bottom line", "tl;dr",
"no explanation needed", "keep it short", "high-level view", etc. NO otherwise)
---
EXECUTION_MODE: EXPLORATORY | PROOF
MODE_REASON: <brief explanation why this mode, max 20 words>
---
CACHED_ANSWER: <answer if question can be answered from known facts, or NONE>
---
