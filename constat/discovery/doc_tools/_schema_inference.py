# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Schema inference for structured data files (CSV, JSON, JSONL)."""

import glob as glob_module
from pathlib import Path
from typing import Optional

from constat.discovery.models import StructuredFileSchema


def _is_glob_pattern(path: str) -> bool:
    """Check if a path contains glob pattern characters."""
    return any(c in path for c in ["*", "?", "[", "]"])


def _expand_file_paths(path: str) -> list[tuple[str, Path]]:
    """
    Expand a path that may be a glob pattern, directory, or single file.

    Args:
        path: File path, glob pattern, or directory path

    Returns:
        List of (display_name, resolved_path) tuples
    """
    p = Path(path)

    # Case 1: Glob pattern
    if _is_glob_pattern(path):
        matches = sorted(glob_module.glob(path, recursive=True))
        return [(Path(m).name, Path(m)) for m in matches if Path(m).is_file()]

    # Case 2: Directory - list all files
    if p.is_dir():
        files = []
        for f in sorted(p.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                files.append((f.name, f))
        return files

    # Case 3: Single file
    if p.exists():
        return [(p.name, p)]

    # Path doesn't exist yet - return as-is for later error handling
    return [(p.name, p)]



def _infer_csv_schema(filepath: Path, sample_rows: int = 100) -> StructuredFileSchema:
    """Infer schema from a CSV file."""
    import csv

    columns = []
    row_count = 0

    with open(filepath, 'r', newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return StructuredFileSchema(
                filename=filepath.name,
                filepath=str(filepath),
                file_format="csv",
                row_count=0,
                columns=[],
            )

        # Initialize column info
        col_data = {h: {'name': h, 'values': []} for h in headers}

        # Sample rows to infer types and collect sample values
        for i, row in enumerate(reader):
            row_count += 1
            if i < sample_rows:
                for j, val in enumerate(row):
                    if j < len(headers):
                        col_data[headers[j]]['values'].append(val)

        # Count remaining rows
        for _ in reader:
            row_count += 1

    # Infer types and get sample values
    for header in headers:
        values = col_data[header]['values']
        col_type = _infer_column_type(values)
        unique_values = list(set(v for v in values if v))[:10]

        columns.append({
            'name': header,
            'type': col_type,
            'sample_values': unique_values,
        })

    return StructuredFileSchema(
        filename=filepath.name,
        filepath=str(filepath),
        file_format="csv",
        row_count=row_count,
        columns=columns,
    )


def _infer_json_schema(filepath: Path, sample_docs: int = 100) -> StructuredFileSchema:
    """Infer schema from a JSON file using DuckDB."""
    from constat.catalog.file._duckdb_inference import infer_json_schema_duckdb

    try:
        result = infer_json_schema_duckdb(str(filepath), sample_size=sample_docs, jsonl=False)
    except Exception:
        return StructuredFileSchema(
            filename=filepath.name,
            filepath=str(filepath),
            file_format="json",
            row_count=0,
            columns=[],
        )

    columns = [
        {"name": c["name"], "type": c["type"], "sample_values": c["sample_values"]}
        for c in result["columns"]
    ]

    return StructuredFileSchema(
        filename=filepath.name,
        filepath=str(filepath),
        file_format="json",
        row_count=result["row_count"],
        columns=columns,
    )


def _infer_column_type(values: list[str]) -> str:
    """Infer column type from string values."""
    if not values:
        return "unknown"

    # Check for numeric types
    int_count = 0
    float_count = 0
    date_count = 0

    for v in values:
        if not v:
            continue
        try:
            int(v)
            int_count += 1
            continue
        except ValueError:
            pass
        try:
            float(v)
            float_count += 1
            continue
        except ValueError:
            pass
        # Simple date check
        if len(v) == 10 and v[4:5] == '-' and v[7:8] == '-':
            date_count += 1

    non_empty = len([v for v in values if v])
    if non_empty == 0:
        return "unknown"

    if int_count == non_empty:
        return "integer"
    if int_count + float_count == non_empty:
        return "float"
    if date_count > non_empty * 0.8:
        return "date"
    return "string"


def _infer_structured_schema(filepath: Path, description: Optional[str] = None) -> Optional[StructuredFileSchema]:
    """Infer schema for a structured file based on its extension."""
    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        schema = _infer_csv_schema(filepath)
    elif suffix == ".json":
        schema = _infer_json_schema(filepath)
    elif suffix == ".jsonl":
        # JSON Lines - read first N lines as separate JSON objects
        schema = _infer_jsonl_schema(filepath)
    else:
        return None

    schema.description = description
    return schema


def _infer_jsonl_schema(filepath: Path, sample_lines: int = 100) -> StructuredFileSchema:
    """Infer schema from a JSON Lines file using DuckDB."""
    from constat.catalog.file._duckdb_inference import infer_json_schema_duckdb

    try:
        result = infer_json_schema_duckdb(str(filepath), sample_size=sample_lines, jsonl=True)
    except Exception:
        return StructuredFileSchema(
            filename=filepath.name,
            filepath=str(filepath),
            file_format="jsonl",
            row_count=0,
            columns=[],
        )

    columns = [
        {"name": c["name"], "type": c["type"], "sample_values": c["sample_values"]}
        for c in result["columns"]
    ]

    return StructuredFileSchema(
        filename=filepath.name,
        filepath=str(filepath),
        file_format="jsonl",
        row_count=result["row_count"],
        columns=columns,
    )
