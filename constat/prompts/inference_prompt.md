Generate Python code for this inference:

Inference: {inf_id}: {inf_name} = {operation}
Explanation: {explanation}

SCALAR VALUES:
{scalars}

TABLES (already in datastore, query with store.query()):
{tables}
{referenced_section}{api_sources_section}
Available Data Access:
{data_source_apis}

CRITICAL Rules:
1. PRESERVE ALL ROWS - Do NOT aggregate to a single row unless explicitly asked
   - For joins: keep all matching rows (use LEFT JOIN to preserve all from left table)
   - For filters: return all rows that match the condition
   - WRONG: Getting only the MAX/MIN/first row
   - RIGHT: Getting all rows that meet criteria
2. Use `if len(df) > 0:` not `if df:` for DataFrame checks
3. End with `_result = <value>`
4. ALWAYS save result: store.save_dataframe('{table_name}', result_df)
5. For REFERENCED database tables, use pd.read_sql(query, db_<name>) with the specific database connection
6. For API SOURCES, use api_<name>() to fetch data, then convert to DataFrame with pd.DataFrame()
   REST APIs often wrap data in paginated responses. Always extract the array:
   ```
   response = api_name('GET /endpoint', {{params}})
   if isinstance(response, dict):
       # Try common wrapper keys
       for key in ['data', 'results', 'items', 'records']:
           if key in response and isinstance(response[key], list):
               response = response[key]
               break
   df = pd.DataFrame(response)
   ```
7. If data isn't in store, and you query the database or API instead, that's normal — NOT an error. But NEVER wrap calls in try/except with hardcoded fallback data. Let errors propagate for the retry mechanism to fix.
   TRUST PRIOR RESULTS: When a dependency table already contains columns you need (reasoning, scores, mappings), use them directly. Do NOT regenerate data that a prior inference already computed and saved. If a dependency is empty or broken, raise an error — do NOT defensively recompute.
8. For ANY VALUE MAPPING, CLASSIFICATION, or DATA EXTRACTION requiring world knowledge:
   NEVER hardcode a mapping dictionary, classification table, or extracted constants.
   Use the appropriate LLM primitive:
   - `llm_map(values, target, source_desc, reason=False, score=False)` — value mapping (names → codes, etc.). ALWAYS returns a best-effort mapping for every input — never null. Returns dict[str, str] by default. With `reason=True`/`score=True`, returns dict[str, dict] with keys "value", "reason", "score". Use score to filter weak matches.
   - `llm_classify(values, categories, context, reason=False, score=False)` — classification into **semantic categories you defined** (e.g., sentiment, priority, risk). NOT for matching to a domain list. Same rich return shape as llm_map when reason/score enabled.
   - `llm_extract(texts, fields, context)` — structured field extraction from free text. `fields` is a list of strings. Returns a dict if one text is passed, list of dicts if multiple.
   - `llm_summarize(texts, instruction)` — text summarization/condensation
   - `llm_score(texts, min_val, max_val, instruction)` — numeric scoring with reasoning. Returns list of `(score, reasoning)` tuples

   Example (mapping):
   mapping = llm_map(df['country'].unique().tolist(), "ISO 3166-1 alpha-2 country code", "country names")
   df['country_iso'] = df['country'].map(mapping)

   Example (classification):
   classes = llm_classify(df['description'].tolist(), ["bug", "feature", "question"], "support tickets")
   df['category'] = df['description'].map(classes)

   Example (scoring):
   results = llm_score(df['review_text'].tolist(), min_val=0.0, max_val=3.0, instruction="Rate sentiment of this employee evaluation")
   df['sentiment_score'] = [score for score, _ in results]
   df['score_reasoning'] = [reason for _, reason in results]

   Hardcoded dicts embed unverifiable LLM knowledge and WILL be flagged.

   PRIMITIVE SELECTION — key distinction:
   - `llm_map` = target values come from a known domain (breeds, countries, ISO codes). LLM generates the value. Output is open-ended.
   - `llm_classify` = you defined semantic categories (high/med/low, bug/feature/question). LLM picks from YOUR labels.
   - WRONG: `llm_classify(items, breed_names)` — breeds are a domain to map to, not semantic categories. Use `llm_map(items, "cat breed")`.
   - WRONG: `llm_classify(cities, country_list)` — use `llm_map(cities, "country")`.
   - RIGHT: `llm_classify(tickets, ["bug", "feature", "question"])` — semantic buckets you defined.
9. MISSING COLUMNS: If an expected column is missing, raise an error listing actual columns:
   `raise KeyError(f"Expected 'col' but columns are: {{list(df.columns)}}")`
   NEVER silently default to zero or skip with `if col in df.columns: ... else: 0`.
   Silent fallbacks produce corrupt data that passes downstream undetected.
10. NEVER use bare `except:` that silently writes empty strings or default values for API calls.
   Let API errors propagate so the retry mechanism can fix the query.
   WRONG: `except: data = ""`
   RIGHT: Let the exception raise, or catch specifically and re-raise with context:
   `except Exception as e: raise ValueError(f"API call failed for {{item}}: {{e}}")`
11. STATIC DATA PROHIBITION: These patterns are FORBIDDEN:
    - Dict literals with 3+ string keys used in .map() — use llm_map()
    - Lists of domain-specific string constants for classification — use llm_classify()
    - Manually constructed extraction results — use llm_extract()
    All LLM knowledge must flow through llm_* primitives for tracking and auditability.
12. DOCUMENT-SOURCED DATA: `doc_read()` is ONLY for configured reference documents listed
    as [DOCUMENT] dependencies in the SCALAR VALUES section above.
    NEVER call doc_read() for data that is in a datastore table — use store.query() instead.
    When business rules, policy thresholds, or configuration values
    come from a reference document, NEVER hardcode them. Load the document at runtime:
    ```
    policy_text = doc_read('compensation_policy')
    rules = llm_extract([policy_text], ['rating_5_raise_pct', 'rating_4_raise_pct', ...], 'raise percentages by rating')
    # rules is a dict: {{'rating_5_raise_pct': '8-12%', 'rating_4_raise_pct': '5-8%', ...}}
    ```
    This ensures code stays current when documents are updated.
13. STRING-TO-NUMERIC CONVERSION: Data from documents, APIs, or mapped values often contains
    string-formatted numbers like "8-12%", "$1,200", "5%". Use the built-in `parse_number()`:
    ```
    # parse_number(val) → tuple of ALL extracted numbers
    # "8-12%" → (8.0, 12.0), "5%" → (5.0,), "1,2,3" → (1.0, 2.0, 3.0)
    # Also: "8 to 12", "$1,200", "10k", "1.5M", "(5%)" → accounting negative
    df['raise_min'] = df['raise_pct'].apply(lambda v: min(parse_number(v)))
    df['raise_max'] = df['raise_pct'].apply(lambda v: max(parse_number(v)))
    ```
    ALWAYS use `parse_number()` for string-to-numeric conversion. NEVER write your own parser.
    NEVER save columns with string-formatted numbers — downstream steps cannot aggregate them.

Return ONLY Python code, no markdown.
