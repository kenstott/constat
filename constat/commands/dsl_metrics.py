# Copyright (c) 2025 Kenneth Stott
# Canary: 0231f950-bf89-4855-8c82-435955c4e8d6
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""CLI: aggregate DSL primitive metrics across all .constat sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _collect_step_rows(root: Path) -> list[dict]:
    rows = []
    for index in root.rglob("steps/index.jsonl"):
        session_dir = index.parent.parent
        session_id = session_dir.name
        user_dir = session_dir.parent.parent
        user_id = user_dir.name
        try:
            for line in index.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec.setdefault("primitive", None)
                rec.setdefault("success", None)
                rec.setdefault("attempts", None)
                rec["session_id"] = session_id
                rec["user_id"] = user_id
                rows.append(rec)
        except Exception:
            pass
    return rows


def _collect_query_rows(root: Path) -> list[dict]:
    rows = []
    for qfile in root.rglob("queries.jsonl"):
        session_dir = qfile.parent
        session_id = session_dir.name
        user_dir = session_dir.parent.parent
        user_id = user_dir.name
        try:
            for line in qfile.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "query_id" not in rec:
                    continue  # skip non-query lines (e.g. "type":"query" preamble)
                rec.setdefault("primitive_retries", 0)
                rec["session_id"] = session_id
                rec["user_id"] = user_id
                rows.append(rec)
        except Exception:
            pass
    return rows


def run(constat_dir: str = ".constat") -> None:
    try:
        import duckdb
    except ImportError:
        print("duckdb not installed", file=sys.stderr)
        sys.exit(1)

    root = Path(constat_dir)
    if not root.exists():
        print(f"{root} not found", file=sys.stderr)
        sys.exit(1)

    steps = _collect_step_rows(root)
    queries = _collect_query_rows(root)

    if not steps and not queries:
        print("No session data found.")
        return

    db = duckdb.connect()

    import pandas as pd
    if steps:
        steps_df = pd.DataFrame(steps)
        # ensure nullable string columns don't get inferred as int
        for col in ("primitive", "model"):
            if col in steps_df.columns:
                steps_df[col] = steps_df[col].astype(str).where(steps_df[col].notna(), None)
        db.register("steps_view", steps_df)
        db.execute("CREATE TABLE steps AS SELECT * FROM steps_view")
    if queries:
        queries_df = pd.DataFrame(queries)
        db.register("queries_view", queries_df)
        db.execute("CREATE TABLE queries AS SELECT * FROM queries_view")

    print("\n=== Per-primitive step success rate ===")
    if steps:
        result = db.execute("""
            SELECT
                COALESCE(primitive, '(unknown)') AS primitive,
                COUNT(*) AS total_steps,
                SUM(CASE WHEN success = true THEN 1 ELSE 0 END) AS succeeded,
                ROUND(100.0 * SUM(CASE WHEN success = true THEN 1 ELSE 0 END) / COUNT(*), 1) AS success_pct,
                ROUND(AVG(CASE WHEN attempts IS NOT NULL THEN attempts ELSE NULL END), 2) AS avg_attempts
            FROM steps
            GROUP BY primitive
            ORDER BY total_steps DESC
        """).fetchall()
        print(f"{'primitive':<20} {'total':>7} {'ok':>6} {'ok%':>6} {'avg_attempts':>13}")
        print("-" * 58)
        for row in result:
            print(f"{row[0]:<20} {row[1]:>7} {row[2]:>6} {row[3]:>6} {row[4] or 'n/a':>13}")
    else:
        print("  No step data with primitive field yet.")

    print("\n=== Plan primitive retry loop activations (per query) ===")
    if queries:
        result = db.execute("""
            SELECT
                SUM(primitive_retries) AS total_retries,
                COUNT(*) AS total_queries,
                SUM(CASE WHEN primitive_retries > 0 THEN 1 ELSE 0 END) AS queries_needing_retry,
                ROUND(100.0 * SUM(CASE WHEN primitive_retries > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS retry_pct
            FROM queries
        """).fetchone()
        print(f"  Total queries:            {result[1]}")
        print(f"  Queries needing retry:    {result[2]} ({result[3]}%)")
        print(f"  Total retry loop fires:   {result[0]}")
    else:
        print("  No query data found.")

    print("\n=== Step attempts distribution by primitive ===")
    if steps:
        result = db.execute("""
            SELECT
                COALESCE(primitive, '(unknown)') AS primitive,
                MIN(attempts) AS min_att,
                MAX(attempts) AS max_att,
                ROUND(AVG(attempts), 2) AS avg_att,
                SUM(CASE WHEN attempts > 1 THEN 1 ELSE 0 END) AS multi_attempt_steps
            FROM steps
            WHERE attempts IS NOT NULL
            GROUP BY primitive
            ORDER BY avg_att DESC
        """).fetchall()
        print(f"{'primitive':<20} {'min':>5} {'max':>5} {'avg':>6} {'retried':>8}")
        print("-" * 50)
        for row in result:
            print(f"{row[0]:<20} {row[1]:>5} {row[2]:>5} {row[3]:>6} {row[4]:>8}")
    else:
        print("  No step data with attempts field yet.")

    db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="DSL primitive metrics across constat sessions")
    parser.add_argument("--dir", default=".constat", help="Path to .constat directory")
    args = parser.parse_args()
    run(args.dir)


if __name__ == "__main__":
    main()
