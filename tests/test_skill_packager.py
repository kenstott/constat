# Copyright (c) 2025 Kenneth Stott
#
# Tests for constat.core.skill_packager

import zipfile
import io

import pytest

from constat.core.skills import SkillManager
from constat.core.skill_packager import (
    package_skill,
    clean_skill_md,
    collect_transitive_dependencies,
    flatten_script,
    detect_imports,
)


def _make_skill(tmp_path, user_id, name, content, script_content=None, refs=None):
    """Helper to create a skill directory with SKILL.md and optional scripts."""
    skills_dir = tmp_path / user_id / "skills" / name
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "SKILL.md").write_text(content)
    if script_content:
        scripts_dir = skills_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        (scripts_dir / "proof.py").write_text(script_content)
    if refs:
        refs_dir = skills_dir / "references"
        refs_dir.mkdir(exist_ok=True)
        for fname, fcontent in refs.items():
            (refs_dir / fname).write_text(fcontent)


# --- clean_skill_md ---

def test_clean_skill_md_strips_internal_fields():
    content = """---
name: my-skill
description: A test skill
exports:
  - script: proof.py
    functions: [run_proof]
dependencies:
  - other-skill
context: fork
agent: Explore
model: sonnet
disable-model-invocation: true
user-invocable: false
argument-hint: "[file]"
allowed-tools: [Read, Grep]
---

Hello world"""

    cleaned = clean_skill_md(content, "my-skill")
    assert "exports" not in cleaned.split("---")[1]
    assert "dependencies" not in cleaned.split("---")[1]
    assert "context" not in cleaned.split("---")[1]
    assert "agent" not in cleaned.split("---")[1]
    assert "model" not in cleaned.split("---")[1]
    assert "disable-model-invocation" not in cleaned.split("---")[1]
    assert "user-invocable" not in cleaned.split("---")[1]
    assert "argument-hint" not in cleaned.split("---")[1]
    assert "allowed-tools" not in cleaned.split("---")[1]
    assert "name: my-skill" in cleaned
    assert "description: A test skill" in cleaned
    assert "Hello world" in cleaned


def test_clean_skill_md_validates_name():
    content = """---
name: My Skill With CAPS!
description: test
---

body"""

    cleaned = clean_skill_md(content, "fallback")
    # Name should be lowercased and invalid chars replaced with hyphens
    parts = cleaned.split("---")
    assert "my-skill-with-caps-" in parts[1]


def test_clean_skill_md_adds_description_if_missing():
    content = """---
name: my-skill
---

body"""

    cleaned = clean_skill_md(content, "my-skill")
    assert "description:" in cleaned


def test_clean_skill_md_no_frontmatter():
    content = "Just a plain markdown file"
    assert clean_skill_md(content, "test") == content


# --- collect_transitive_dependencies ---

def test_collect_transitive_dependencies(tmp_path):
    _make_skill(tmp_path, "u1", "root-skill", """---
name: root-skill
description: root
dependencies:
  - dep-a
---

Root body""")

    _make_skill(tmp_path, "u1", "dep-a", """---
name: dep-a
description: dep a
dependencies:
  - dep-b
exports:
  - script: proof.py
    functions: [helper_a]
---

Dep A body""", script_content="def helper_a(): pass")

    _make_skill(tmp_path, "u1", "dep-b", """---
name: dep-b
description: dep b
exports:
  - script: proof.py
    functions: [helper_b]
---

Dep B body""", script_content="def helper_b(): pass")

    mgr = SkillManager("u1", base_dir=tmp_path)
    deps = collect_transitive_dependencies("root-skill", mgr)
    dep_names = [d.name for d in deps]
    assert "dep-a" in dep_names
    assert "dep-b" in dep_names
    assert "root-skill" not in dep_names


def test_collect_no_dependencies(tmp_path):
    _make_skill(tmp_path, "u1", "standalone", """---
name: standalone
description: no deps
---

body""")

    mgr = SkillManager("u1", base_dir=tmp_path)
    assert collect_transitive_dependencies("standalone", mgr) == []


# --- flatten_script ---

def test_flatten_script(tmp_path):
    _make_skill(tmp_path, "u1", "helper-lib", """---
name: helper-lib
description: helper
exports:
  - script: proof.py
    functions: [compute]
---

body""", script_content="""
def compute(x):
    return x * 2

def _private():
    pass
""")

    mgr = SkillManager("u1", base_dir=tmp_path)
    dep_skill = mgr.get_skill("helper-lib")
    root_source = "result = helper_lib_compute(42)\nprint(result)"
    flat = flatten_script(root_source, [dep_skill], mgr)

    assert "def helper_lib_compute" in flat
    assert "result = helper_lib_compute(42)" in flat
    # _private should NOT be inlined (not in exports)
    assert "_private" not in flat


def test_flatten_script_no_deps():
    source = "print('hello')"
    assert flatten_script(source, [], None) == source


# --- detect_imports ---

def test_detect_imports_finds_non_preinstalled():
    source = """
import duckdb
import pandas as pd
from pathlib import Path
import some_custom_lib
"""
    deps = detect_imports(source)
    assert "duckdb" in deps
    assert "some_custom_lib" in deps
    assert "pandas" not in deps
    assert "pathlib" not in deps


def test_detect_imports_maps_pip_names():
    source = "import yaml\nimport dateutil"
    deps = detect_imports(source)
    assert "pyyaml" in deps
    assert "python-dateutil" in deps


def test_detect_imports_empty():
    assert detect_imports("x = 1") == []


def test_detect_imports_syntax_error():
    assert detect_imports("def broken(") == []


# --- package_skill (integration) ---

def test_package_skill_basic(tmp_path):
    _make_skill(tmp_path, "u1", "test-pkg", """---
name: test-pkg
description: Package test
exports:
  - script: proof.py
    functions: [run_proof]
allowed-tools: [Read]
context: fork
---

Instructions here""",
        script_content="import pandas\ndef run_proof():\n    return {}\n",
        refs={"guide.md": "# Guide\nSome reference content"},
    )

    mgr = SkillManager("u1", base_dir=tmp_path)
    zip_bytes = package_skill("test-pkg", mgr)
    assert len(zip_bytes) > 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert "test-pkg/SKILL.md" in names
        assert "test-pkg/scripts/proof.py" in names
        assert "test-pkg/references/guide.md" in names

        # SKILL.md should have internal fields stripped
        skill_md = zf.read("test-pkg/SKILL.md").decode()
        assert "exports" not in skill_md.split("---")[1]
        assert "allowed-tools" not in skill_md.split("---")[1]
        assert "context" not in skill_md.split("---")[1]
        assert "Instructions here" in skill_md


def test_package_skill_not_found(tmp_path):
    mgr = SkillManager("u1", base_dir=tmp_path)
    with pytest.raises(ValueError, match="Skill not found"):
        package_skill("nonexistent", mgr)


def test_package_skill_with_dependencies(tmp_path):
    _make_skill(tmp_path, "u1", "main-skill", """---
name: main-skill
description: Main
dependencies:
  - util-skill
---

Main body""",
        script_content="result = util_skill_add(1, 2)\n",
    )

    _make_skill(tmp_path, "u1", "util-skill", """---
name: util-skill
description: Utils
exports:
  - script: proof.py
    functions: [add]
---

Utils body""",
        script_content="def add(a, b):\n    return a + b\n",
    )

    mgr = SkillManager("u1", base_dir=tmp_path)
    zip_bytes = package_skill("main-skill", mgr)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        proof = zf.read("main-skill/scripts/proof.py").decode()
        assert "def util_skill_add" in proof
        assert "result = util_skill_add(1, 2)" in proof
