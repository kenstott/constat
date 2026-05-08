# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Hash-cache bridge using DuckDB.

Provides get/set/delete for a doc_hashes table keyed by (namespace, document_name).
Accepts a DuckDB connection object — the caller passes it in.
"""


def _ensure_table(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS doc_hashes (
            namespace TEXT NOT NULL,
            document_name TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            PRIMARY KEY (namespace, document_name)
        )
    """)


def get_hash(conn, namespace: str, document_name: str) -> str | None:
    _ensure_table(conn)
    row = conn.execute(
        "SELECT content_hash FROM doc_hashes WHERE namespace = ? AND document_name = ?",
        [namespace, document_name],
    ).fetchone()
    return row[0] if row else None


def set_hash(conn, namespace: str, document_name: str, content_hash: str) -> None:
    _ensure_table(conn)
    conn.execute("""
        INSERT INTO doc_hashes (namespace, document_name, content_hash) VALUES (?, ?, ?)
        ON CONFLICT (namespace, document_name) DO UPDATE SET content_hash = excluded.content_hash
    """, [namespace, document_name, content_hash])


def delete_hash(conn, namespace: str, document_name: str) -> bool:
    _ensure_table(conn)
    row = conn.execute(
        "DELETE FROM doc_hashes WHERE namespace = ? AND document_name = ? RETURNING namespace",
        [namespace, document_name],
    ).fetchone()
    return row is not None
