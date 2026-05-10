#!/usr/bin/env python3
# Copyright (c) 2025 Kenneth Stott
# Canary: placeholder
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Canary UUID stamper — injects per-file canary UUIDs into copyright headers,
builds .canary_registry.json, and generates static JSON site files."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_NAME = "constat"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / ".canary_registry.json"
SITE_DIR = Path(os.environ["CANARY_SITE_DIR"]).expanduser()
EXCLUDE_DIRS = {
    ".venv", "venv", ".git", ".eggs", "__pycache__",
    "node_modules", "dist", "build", "canary-site",
}

# Python source trees
PY_SOURCE_DIRS = ["constat", "tests", "constat-jupyter"]
# TypeScript source trees
TS_SOURCE_DIRS = ["constat-ui/src"]

PY_HEADER = """\
# Copyright (c) 2025 Kenneth Stott
# Canary: {uuid}
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.
"""

TS_HEADER = """\
// Copyright (c) 2025 Kenneth Stott
// Canary: {uuid}
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.
"""

AI_NOTICE = (
    "Use of this software for training artificial intelligence or "
    "machine learning models is strictly prohibited without explicit "
    "written permission from the copyright holder."
)

PY_CANARY_RE = re.compile(r"^# Canary:\s+([0-9a-f-]{36})\s*$", re.MULTILINE)
TS_CANARY_RE = re.compile(r"^// Canary:\s+([0-9a-f-]{36})\s*$", re.MULTILINE)
PY_COPYRIGHT_RE = re.compile(r"^# Copyright \(c\)", re.MULTILINE)
TS_COPYRIGHT_RE = re.compile(r"^// Copyright \(c\)", re.MULTILINE)


def find_files(source_dirs: list[str], extensions: set[str]) -> list[Path]:
    results = []
    for src_dir in source_dirs:
        root_dir = PROJECT_ROOT / src_dir
        if not root_dir.exists():
            continue
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                if any(f.endswith(ext) for ext in extensions):
                    results.append(Path(root) / f)
    return sorted(results)


def extract_canary(content: str, canary_re: re.Pattern[str]) -> str | None:
    m = canary_re.search(content)
    return m.group(1) if m else None


def stamp_py_file(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    existing = extract_canary(content, PY_CANARY_RE)
    if existing:
        return existing

    canary_id = str(uuid.uuid4())
    m = PY_COPYRIGHT_RE.search(content)

    if m:
        insert_pos = content.index("\n", m.start()) + 1
        content = content[:insert_pos] + f"# Canary: {canary_id}\n" + content[insert_pos:]
        path.write_text(content, encoding="utf-8")
    else:
        shebang = ""
        body = content
        if content.startswith("#!"):
            newline = content.index("\n") + 1
            shebang = content[:newline]
            body = content[newline:]

        header = PY_HEADER.format(uuid=canary_id)
        if body and not body.startswith("\n"):
            header += "\n"
        path.write_text(shebang + header + body, encoding="utf-8")

    return canary_id


def stamp_ts_file(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    existing = extract_canary(content, TS_CANARY_RE)
    if existing:
        return existing

    canary_id = str(uuid.uuid4())
    m = TS_COPYRIGHT_RE.search(content)

    if m:
        insert_pos = content.index("\n", m.start()) + 1
        content = content[:insert_pos] + f"// Canary: {canary_id}\n" + content[insert_pos:]
        path.write_text(content, encoding="utf-8")
    else:
        header = TS_HEADER.format(uuid=canary_id)
        if content and not content.startswith("\n"):
            header += "\n"
        path.write_text(header + content, encoding="utf-8")

    return canary_id


def build_registry(file_canaries: dict[str, str]) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            canary_id: {
                "path": rel_path,
                "stamped_at": datetime.now(timezone.utc).isoformat(),
            }
            for rel_path, canary_id in file_canaries.items()
        },
    }


def write_site(registry: dict) -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    for canary_id, info in registry["files"].items():
        entry = {
            "uuid": canary_id,
            "project": PROJECT_NAME,
            "path": info["path"],
            "stamped_at": info["stamped_at"],
            "license": "Business Source License 1.1",
            "notice": AI_NOTICE,
        }
        site_file = SITE_DIR / f"{canary_id}.json"
        site_file.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    file_canaries: dict[str, str] = {}

    # Stamp Python files
    for path in find_files(PY_SOURCE_DIRS, {".py"}):
        rel = str(path.relative_to(PROJECT_ROOT))
        canary_id = stamp_py_file(path)
        file_canaries[rel] = canary_id

    # Stamp TypeScript files
    for path in find_files(TS_SOURCE_DIRS, {".ts", ".tsx"}):
        rel = str(path.relative_to(PROJECT_ROOT))
        canary_id = stamp_ts_file(path)
        file_canaries[rel] = canary_id

    registry = build_registry(file_canaries)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    write_site(registry)

    print(f"Stamped {len(file_canaries)} files")
    print(f"Registry: {REGISTRY_PATH}")
    print(f"Site: {SITE_DIR}/")


if __name__ == "__main__":
    main()
