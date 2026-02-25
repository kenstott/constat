# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""DuckDB-based schema inference for JSON/JSONL files."""

import duckdb

# DuckDB type prefix -> our type string
_TYPE_MAP = {
    "VARCHAR": "string",
    "BIGINT": "integer",
    "INTEGER": "integer",
    "SMALLINT": "integer",
    "TINYINT": "integer",
    "HUGEINT": "integer",
    "UBIGINT": "integer",
    "UINTEGER": "integer",
    "USMALLINT": "integer",
    "UTINYINT": "integer",
    "DOUBLE": "float",
    "FLOAT": "float",
    "DECIMAL": "float",
    "BOOLEAN": "boolean",
    "TIMESTAMP": "timestamp",
    "TIMESTAMP WITH TIME ZONE": "timestamp",
    "DATE": "date",
    "TIME": "time",
    "BLOB": "binary",
    "JSON": "string",
    "UUID": "string",
}


def _map_duckdb_type(duckdb_type: str) -> str:
    """Map a DuckDB type string to our normalized type string."""
    upper = duckdb_type.upper()
    # Exact match first
    if upper in _TYPE_MAP:
        return _TYPE_MAP[upper]
    # Prefix match for parameterized types like DECIMAL(18,3)
    for prefix, mapped in _TYPE_MAP.items():
        if upper.startswith(prefix):
            return mapped
    # Struct/list/map
    if upper.startswith("STRUCT"):
        return "object"
    if upper.endswith("[]") or upper.startswith("LIST"):
        return "array"
    if upper.startswith("MAP"):
        return "map"
    return "string"


def infer_json_schema_duckdb(
    path: str,
    sample_size: int = 100,
    jsonl: bool = False,
) -> dict:
    """Infer schema from a JSON or JSONL file using DuckDB.

    Returns:
        {
            "columns": [{"name": str, "type": str, "nullable": bool, "sample_values": list}],
            "row_count": int,
        }
    """
    conn = duckdb.connect()
    try:
        fmt = "newline_delimited" if jsonl else "auto"
        read_expr = f"read_json_auto('{path}', format='{fmt}')"

        # Get column names and types
        describe = conn.sql(f"DESCRIBE SELECT * FROM {read_expr}").fetchall()

        # Get sample values
        sample = conn.sql(f"SELECT * FROM {read_expr} LIMIT {sample_size}").fetchdf()

        # Get row count
        row_count = conn.sql(f"SELECT count(*) FROM {read_expr}").fetchone()[0]

        columns = []
        for col_name, col_type, *_ in describe:
            nullable = True
            sample_values = []
            if col_name in sample.columns:
                series = sample[col_name].dropna()
                raw = series.head(10).tolist()
                # Convert to simple types for display
                for v in raw:
                    if isinstance(v, (str, int, float, bool)):
                        sample_values.append(v)
                    else:
                        sample_values.append(str(v))
                # Deduplicate
                seen = set()
                unique = []
                for v in sample_values:
                    key = (type(v).__name__, v)
                    if key not in seen:
                        seen.add(key)
                        unique.append(v)
                sample_values = unique[:10]

                nullable = bool(sample[col_name].isna().any())

            columns.append({
                "name": col_name,
                "type": _map_duckdb_type(col_type),
                "nullable": nullable,
                "sample_values": sample_values,
            })

        return {"columns": columns, "row_count": row_count}
    finally:
        conn.close()
