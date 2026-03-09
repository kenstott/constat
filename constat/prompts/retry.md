Your previous code failed to execute.

{error_details}

Previous code:
```python
{previous_code}
```
{data_sources}
Fix the error and try again. Return ONLY the corrected Python code wrapped in ```python ... ``` markers.

CRITICAL RULES FOR RETRIES:
- Fix the ACTUAL root cause (wrong column name, wrong table, wrong key).
- NEVER add try/except. Not for any reason. No error handling. Fix the bug, do not catch it.
- NEVER invent default values (percentages, budgets, thresholds, ratings) as a fallback for a failed call.
- NEVER use `.get('key', hardcoded_default)` to hide missing keys. Access keys directly.
- NEVER use `fillna()` with invented business values (e.g. `fillna(3.0)` for ratings).
- If a column doesn't exist, query the actual schema first (`SELECT * FROM table LIMIT 1`) then use the correct column names.
- If `doc_read()` or `llm_extract()` failed, fix the call — don't replace it with hardcoded data.
- If a table is not found, use the `find_relevant_tables` or `find_entity` tools to locate it — do NOT guess table names or locations.
- `store` only contains intermediate tables saved by previous steps. Source data lives in database connections (`db_<name>`).