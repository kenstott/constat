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
5. For REFERENCED database tables, use db_query() or pd.read_sql(query, db_<name>)
6. For API SOURCES, use api_<name>() to fetch data, then convert to DataFrame with pd.DataFrame()
7. Don't label expected fallbacks as errors - querying a database or API when data isn't in store is normal
8. For FUZZY MAPPING (e.g., free-text names → codes): first try the data source (API/database).
   For values that can't be matched exactly, use `llm_map(values, target, source_desc)` as fallback:
   ```
   unmatched = [name for name in names if name not in exact_matches]
   fuzzy = llm_map(unmatched, "ISO 3166-1 alpha-2 country code", "country names")
   ```
   This reduces proof confidence — use only when data sources can't resolve the mapping.
9. NEVER use bare `except:` that silently writes empty strings or default values for API calls.
   Let API errors propagate so the retry mechanism can fix the query.
   WRONG: `except: data = ""`
   RIGHT: Let the exception raise, or catch specifically and re-raise with context:
   `except Exception as e: raise ValueError(f"API call failed for {{item}}: {{e}}")`

Return ONLY Python code, no markdown.