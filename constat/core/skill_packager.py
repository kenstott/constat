# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Package a skill as a standalone zip for Claude Desktop / Claude Code."""

import ast
import io
import logging
import re
import zipfile
from typing import TYPE_CHECKING

import yaml

from constat.core.skills import parse_frontmatter

if TYPE_CHECKING:
    from constat.core.skills import Skill, SkillManager

logger = logging.getLogger(__name__)

# Frontmatter fields that are internal to constat and should be stripped
_INTERNAL_FIELDS = frozenset({
    "exports",
    "dependencies",
    "context",
    "agent",
    "model",
    "disable-model-invocation",
    "user-invocable",
    "argument-hint",
    "allowed-tools",
})

# Packages pre-installed in Claude Desktop / Claude Code environments
_PREINSTALLED_PACKAGES = frozenset({
    "pandas", "numpy", "scipy", "pyarrow", "matplotlib", "seaborn",
    "sklearn", "scikit-learn", "sqlite3", "json", "csv", "re", "os",
    "sys", "math", "datetime", "collections", "itertools", "functools",
    "pathlib", "typing", "dataclasses", "abc", "io", "logging",
    "hashlib", "base64", "uuid", "copy", "textwrap", "string",
    "operator", "contextlib", "warnings", "statistics", "decimal",
    "fractions", "random", "time", "calendar", "struct", "enum",
    "urllib", "http", "email", "html", "xml", "ftplib", "imaplib",
    "smtplib", "socketserver", "xmlrpc", "ipaddress", "ssl", "socket",
    "select", "selectors", "signal", "mmap", "codecs", "unicodedata",
    "locale", "gettext", "bisect", "heapq", "array", "weakref",
    "types", "pprint", "reprlib", "traceback", "gc", "inspect",
    "dis", "pickletools", "pickle", "shelve", "dbm", "gzip", "bz2",
    "lzma", "zipfile", "tarfile", "tempfile", "glob", "fnmatch",
    "shutil", "fileinput", "filecmp", "stat", "posixpath",
    "genericpath", "ntpath", "linecache", "tokenize", "token",
    "keyword", "difflib", "pydoc", "doctest", "unittest",
    "subprocess", "threading", "multiprocessing", "concurrent",
    "asyncio", "queue", "sched", "_thread", "contextvars",
    "requests", "httpx", "aiohttp", "PIL", "pillow",
    "sympy", "networkx", "statsmodels", "plotly", "bokeh",
    "beautifulsoup4", "bs4", "lxml", "openpyxl", "xlsxwriter",
})

# Map import names to pip package names where they differ
_IMPORT_TO_PIP = {
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "attr": "attrs",
    "dateutil": "python-dateutil",
}


def package_skill(skill_name: str, skill_manager: "SkillManager") -> bytes:
    """Package a skill as a zip file.

    Returns zip bytes containing SKILL.md, scripts/, references/, assets/,
    and requirements.txt.
    """
    skill = skill_manager.get_skill(skill_name)
    if not skill:
        raise ValueError(f"Skill not found: {skill_name}")

    skill_dir = skill_manager.get_skill_dir(skill_name)
    if not skill_dir:
        raise ValueError(f"Skill directory not found: {skill_name}")

    # Read SKILL.md
    skill_file = skill_dir / "SKILL.md"
    raw_content = skill_file.read_text()

    # Clean frontmatter
    cleaned_md = clean_skill_md(raw_content, skill_name)

    # Collect dependencies for script flattening
    deps = collect_transitive_dependencies(skill_name, skill_manager)

    # Read and flatten proof.py if it exists
    scripts_dir = skill_dir / "scripts"
    flat_script = None
    if scripts_dir.exists() and (scripts_dir / "proof.py").exists():
        root_source = (scripts_dir / "proof.py").read_text()
        flat_script = flatten_script(root_source, deps, skill_manager)

    # Build zip
    buf = io.BytesIO()
    prefix = skill.filename or skill_name
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}/SKILL.md", cleaned_md)

        if flat_script:
            zf.writestr(f"{prefix}/scripts/proof.py", flat_script)
            # requirements.txt from flattened script
            extra_deps = detect_imports(flat_script)
            if extra_deps:
                zf.writestr(f"{prefix}/requirements.txt", "\n".join(sorted(extra_deps)) + "\n")

        # Copy references/ and assets/ directories
        for subdir_name in ("references", "assets"):
            subdir = skill_dir / subdir_name
            if subdir.exists() and subdir.is_dir():
                for file_path in subdir.rglob("*"):
                    if file_path.is_file():
                        arcname = f"{prefix}/{subdir_name}/{file_path.relative_to(subdir)}"
                        zf.write(file_path, arcname)

    return buf.getvalue()


def clean_skill_md(content: str, skill_name: str) -> str:
    """Strip internal frontmatter fields and validate Claude constraints.

    Validates:
    - name: max 64 chars, lowercase + hyphens only
    - description: non-empty, max 1024 chars
    """
    frontmatter, body = parse_frontmatter(content)
    if not frontmatter:
        return content

    # Strip internal fields
    cleaned = {k: v for k, v in frontmatter.items() if k not in _INTERNAL_FIELDS}

    # Validate / fix name
    name = cleaned.get("name", skill_name)
    name = re.sub(r"[^a-z0-9-]", "-", str(name).lower())[:64]
    cleaned["name"] = name

    # Validate description
    desc = cleaned.get("description", "")
    if not desc:
        cleaned["description"] = f"Skill: {name}"
    elif len(desc) > 1024:
        cleaned["description"] = desc[:1024]

    fm_yaml = yaml.dump(cleaned, default_flow_style=False, sort_keys=False)
    return f"---\n{fm_yaml}---\n\n{body}\n"


def collect_transitive_dependencies(skill_name: str, skill_manager: "SkillManager") -> list["Skill"]:
    """Collect all transitive dependency skills (not including the root)."""
    skill = skill_manager.get_skill(skill_name)
    if not skill:
        return []

    visited: dict[str, "Skill"] = {}

    def _collect(s: "Skill") -> None:
        if s.name in visited:
            return
        visited[s.name] = s
        for dep_name in s.dependencies:
            dep = skill_manager.get_skill(dep_name)
            if dep:
                _collect(dep)

    for dep_name in skill.dependencies:
        dep = skill_manager.get_skill(dep_name)
        if dep:
            _collect(dep)

    return list(visited.values())


def flatten_script(root_source: str, dependencies: list["Skill"], skill_manager: "SkillManager") -> str:
    """Inline dependency functions into the root script.

    For each dependency skill, extracts exported function defs via ast,
    renames them to {pkg_name}_{fn_name}, and prepends to root script.
    """
    if not dependencies:
        return root_source

    prepended_blocks: list[str] = []

    for dep_skill in dependencies:
        if not dep_skill.exports:
            continue

        dep_dir = skill_manager.get_skill_dir(dep_skill.name)
        if not dep_dir:
            continue

        pkg_name = dep_skill.name.replace("-", "_").replace(" ", "_")

        for export_entry in dep_skill.exports:
            script_name = export_entry.get("script", "")
            fn_names = set(export_entry.get("functions", []))
            if not script_name or not fn_names:
                continue

            script_path = dep_dir / "scripts" / script_name
            if not script_path.exists():
                continue

            source = script_path.read_text()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                logger.warning(f"Failed to parse {script_path}")
                continue

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in fn_names:
                        original_name = node.name
                        node.name = f"{pkg_name}_{original_name}"
                        prepended_blocks.append(ast.unparse(node))

    if not prepended_blocks:
        return root_source

    header = "# --- Inlined dependency functions ---\n"
    return header + "\n\n".join(prepended_blocks) + "\n\n# --- Main script ---\n" + root_source


def detect_imports(script_source: str) -> list[str]:
    """Detect non-preinstalled imports and return pip package names."""
    try:
        tree = ast.parse(script_source)
    except SyntaxError:
        return []

    top_level_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level_modules.add(node.module.split(".")[0])

    deps: list[str] = []
    for mod in sorted(top_level_modules):
        if mod in _PREINSTALLED_PACKAGES:
            continue
        pip_name = _IMPORT_TO_PIP.get(mod, mod)
        if pip_name not in _PREINSTALLED_PACKAGES:
            deps.append(pip_name)
    return deps
