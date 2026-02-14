# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""User skills for customizing system prompts.

Skills follow the Agent Skills open standard (https://agentskills.io).
Each skill is a directory containing a SKILL.md file with YAML frontmatter.

Directory structure:
    {base_dir}/{user_id}/skills/{skill-name}/
    ├── SKILL.md (required)
    ├── scripts/     # Optional executable code
    ├── references/  # Optional documentation
    └── assets/      # Optional templates, icons, etc.

SKILL.md format:
    ---
    name: skill-name
    description: What this skill does
    allowed-tools: [Read, Grep]
    disable-model-invocation: false
    user-invocable: true
    context: fork
    agent: Explore
    model: sonnet
    argument-hint: [filename]
    exports:
      - script: proof.py
        functions: [run_proof, helper_fn]
    dependencies:
      - other-skill-name
    ---

    Markdown instructions here...
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"


def get_skills_dir(user_id: str, base_dir: Optional[Path] = None) -> Path:
    """Get the skills directory for a user.

    Args:
        user_id: User identifier
        base_dir: Base .constat directory (defaults to ./.constat)
    """
    if base_dir is None:
        base_dir = Path(".constat")
    return base_dir / user_id / "skills"


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: Markdown content with optional YAML frontmatter

    Returns:
        Tuple of (frontmatter dict, body content)
    """
    # Check for frontmatter delimiter
    if not content.startswith("---"):
        return {}, content

    # Find the closing delimiter
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
    if not match:
        return {}, content

    frontmatter_text = match.group(1)
    body = match.group(2)

    try:
        frontmatter = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        frontmatter = {}

    return frontmatter, body.strip()


@dataclass
class Skill:
    """A user-defined skill following the Agent Skills standard.

    See https://agentskills.io/specification for the full specification.
    """
    # Required fields
    name: str
    prompt: str  # The markdown body (instructions)

    # Recommended fields
    description: str = ""

    # Optional metadata
    filename: str = ""  # The directory name (skill-name)
    allowed_tools: list[str] = field(default_factory=list)

    # Invocation control
    disable_model_invocation: bool = False  # If true, only user can invoke
    user_invocable: bool = True  # If false, hidden from / menu

    # Execution context
    context: str = ""  # "fork" to run in subagent
    agent: str = ""  # Subagent type when context=fork (e.g., "Explore", "Plan")
    model: str = ""  # Model to use when skill is active

    # UI hints
    argument_hint: str = ""  # Hint for autocomplete (e.g., "[issue-number]")

    # Script exports — declares which functions to inject from which scripts
    # Format: [{"script": "proof.py", "functions": ["run_proof", "parse_number"]}, ...]
    exports: list[dict] = field(default_factory=list)

    # Skill dependencies — names of other skills whose exports this skill calls
    dependencies: list[str] = field(default_factory=list)


class SkillManager:
    """Manages skills loaded from system, project, and user directories.

    Precedence order (later overrides earlier by name):
        1. System:  {config_dir}/skills/  (alongside config.yaml)
        2. Project: {project_dir}/skills/ (for each active project)
        3. User:    {base_dir}/{user_id}/skills/
    """

    def __init__(self, user_id: str = "default", base_dir: Optional[Path] = None,
                 system_skills_dir: Optional[Path] = None):
        """Initialize the skill manager.

        Args:
            user_id: User identifier
            base_dir: Base .constat directory. Defaults to ./.constat
            system_skills_dir: System skills directory (config_dir/skills/).
        """
        self._user_id = user_id
        self._base_dir = base_dir or Path(".constat")
        self._skills_dir = get_skills_dir(user_id, self._base_dir)
        self._system_skills_dir = system_skills_dir
        self._project_skill_dirs: list[Path] = []
        self._skills: dict[str, Skill] = {}
        self._active_skills: set[str] = set()
        self._ensure_skills_dir()
        self._load_skills()

    def _ensure_skills_dir(self) -> None:
        """Ensure the user skills directory exists."""
        self._skills_dir.mkdir(parents=True, exist_ok=True)

    def _load_skills_from_dir(self, skills_dir: Path, source: str) -> None:
        """Load skills from a single directory, overriding existing entries by name."""
        if not skills_dir.exists():
            return

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / SKILL_FILENAME
            if not skill_file.exists():
                continue

            try:
                with open(skill_file, "r") as f:
                    content = f.read()

                frontmatter, body = parse_frontmatter(content)

                name = frontmatter.get("name", skill_dir.name)
                description = frontmatter.get("description", "").strip()
                prompt = body.strip()

                allowed_tools = frontmatter.get("allowed-tools", [])
                disable_model_invocation = frontmatter.get("disable-model-invocation", False)
                user_invocable = frontmatter.get("user-invocable", True)
                context = frontmatter.get("context", "")
                agent = frontmatter.get("agent", "")
                model = frontmatter.get("model", "")
                argument_hint = frontmatter.get("argument-hint", "")
                exports = frontmatter.get("exports", [])
                dependencies = frontmatter.get("dependencies", [])

                if prompt:
                    self._skills[name] = Skill(
                        name=name,
                        prompt=prompt,
                        description=description,
                        filename=skill_dir.name,
                        allowed_tools=allowed_tools or [],
                        disable_model_invocation=disable_model_invocation,
                        user_invocable=user_invocable,
                        context=context,
                        agent=agent,
                        model=model,
                        argument_hint=argument_hint,
                        exports=exports if isinstance(exports, list) else [],
                        dependencies=dependencies if isinstance(dependencies, list) else [],
                    )
                    logger.debug(f"Loaded skill: {name} from {skill_dir.name}/SKILL.md ({source})")

            except Exception as e:
                logger.warning(f"Failed to load skill from {skill_file}: {e}")

    def _load_skills(self) -> None:
        """Load skills from all directories in precedence order.

        System < project < user (last wins).
        """
        self._skills.clear()

        # 1. System skills (lowest precedence) - alongside config.yaml
        if self._system_skills_dir:
            self._load_skills_from_dir(self._system_skills_dir, "system")

        # 2. Active project skill dirs
        for project_dir in self._project_skill_dirs:
            self._load_skills_from_dir(project_dir, "project")

        # 3. User skills (highest precedence)
        self._load_skills_from_dir(self._skills_dir, "user")

        logger.info(f"Loaded {len(self._skills)} skills (system={self._system_skills_dir}, projects={len(self._project_skill_dirs)}, user={self._skills_dir})")

    def add_project_skills(self, project_dir: Path) -> None:
        """Add a project skills directory and reload.

        Args:
            project_dir: Path to the project's skills/ directory.
        """
        if project_dir not in self._project_skill_dirs:
            self._project_skill_dirs.append(project_dir)
            self._load_skills()

    def remove_project_skills(self, project_dir: Path) -> None:
        """Remove a project skills directory and reload."""
        if project_dir in self._project_skill_dirs:
            self._project_skill_dirs.remove(project_dir)
            self._load_skills()

    def reload(self) -> None:
        """Reload skills from files."""
        self._load_skills()

    def list_skills(self) -> list[str]:
        """Get list of available skill names."""
        return list(self._skills.keys())

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_all_skills(self) -> list[Skill]:
        """Get all skills."""
        return list(self._skills.values())

    def activate_skill(self, name: str) -> bool:
        """Activate a skill.

        Args:
            name: Skill name

        Returns:
            True if skill was activated, False if not found
        """
        if name not in self._skills:
            return False
        self._active_skills.add(name)
        return True

    def deactivate_skill(self, name: str) -> bool:
        """Deactivate a skill.

        Args:
            name: Skill name

        Returns:
            True if skill was deactivated, False if wasn't active
        """
        if name in self._active_skills:
            self._active_skills.discard(name)
            return True
        return False

    def set_active_skills(self, names: list[str]) -> list[str]:
        """Set the active skills.

        Args:
            names: List of skill names to activate

        Returns:
            List of skill names that were successfully activated
        """
        self._active_skills.clear()
        activated = []
        for name in names:
            if name in self._skills:
                self._active_skills.add(name)
                activated.append(name)
        return activated

    @property
    def active_skills(self) -> list[str]:
        """Get list of active skill names."""
        return list(self._active_skills)

    @property
    def active_skill_objects(self) -> list[Skill]:
        """Get list of active Skill objects."""
        return [self._skills[name] for name in self._active_skills if name in self._skills]

    def get_skills_prompt(self) -> str:
        """Get combined prompt from all active skills."""
        prompts = []
        for name in sorted(self._active_skills):
            skill = self._skills.get(name)
            if skill:
                prompts.append(f"## {skill.name}\n{skill.prompt}")
        return "\n\n".join(prompts)

    @property
    def has_skills(self) -> bool:
        """Check if any skills are defined."""
        return len(self._skills) > 0

    @property
    def skills_dir(self) -> Path:
        """Get the path to the 'skills' directory."""
        return self._skills_dir

    def get_skill_dir(self, name: str) -> Optional[Path]:
        """Get the full directory path for a skill, searching all skill directories."""
        skill = self._skills.get(name)
        if not skill:
            return None
        # Search in reverse precedence: user > project > system (first match wins)
        search_dirs = [self._skills_dir]
        search_dirs.extend(reversed(self._project_skill_dirs))
        if self._system_skills_dir:
            search_dirs.append(self._system_skills_dir)
        for search_dir in search_dirs:
            candidate = search_dir / skill.filename
            if candidate.exists():
                return candidate
        return None

    # CRUD operations for skills

    def create_skill(self, name: str, prompt: str, description: str = "") -> Skill:
        """Create a new skill.

        Args:
            name: Skill name (will be used as directory name)
            prompt: The skill prompt content
            description: Optional description

        Returns:
            The created Skill object

        Raises:
            ValueError: If skill with this name already exists
        """
        # Sanitize name for directory
        safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in name.lower())
        skill_dir = self._skills_dir / safe_name

        if skill_dir.exists():
            raise ValueError(f"Skill '{name}' already exists")

        # Create directory and SKILL.md file
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / SKILL_FILENAME

        # Build markdown content with frontmatter
        content = f"""---
name: {name}
description: {description}
allowed-tools: []
---

{prompt}
"""

        with open(skill_file, "w") as f:
            f.write(content)

        skill = Skill(
            name=name,
            prompt=prompt,
            description=description,
            filename=safe_name,
        )
        self._skills[name] = skill
        return skill

    def update_skill(self, name: str, prompt: Optional[str] = None,
                     description: Optional[str] = None, new_name: Optional[str] = None) -> bool:
        """Update an existing skill.

        Args:
            name: Current skill name
            prompt: New prompt content (optional)
            description: New description (optional)
            new_name: New name for the skill (optional)

        Returns:
            True if updated successfully, False if skill not found
        """
        skill = self._skills.get(name)
        if not skill:
            return False

        skill_file = self._skills_dir / skill.filename / SKILL_FILENAME

        # Read current content
        with open(skill_file, "r") as f:
            content = f.read()

        frontmatter, body = parse_frontmatter(content)

        # Update fields
        if prompt is not None:
            body = prompt
            skill.prompt = prompt
        if description is not None:
            frontmatter["description"] = description
            skill.description = description
        if new_name is not None and new_name != name:
            frontmatter["name"] = new_name
            # Update in-memory
            del self._skills[name]
            skill.name = new_name
            self._skills[new_name] = skill
            # Update active skills if needed
            if name in self._active_skills:
                self._active_skills.discard(name)
                self._active_skills.add(new_name)

        # Build new content
        frontmatter_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
        new_content = f"---\n{frontmatter_yaml}---\n\n{body}\n"

        # Save back
        with open(skill_file, "w") as f:
            f.write(new_content)

        return True

    def update_skill_content(self, name: str, content: str) -> bool:
        """Update a skill from raw markdown content.

        Args:
            name: Skill name
            content: Raw markdown content with YAML frontmatter

        Returns:
            True if updated successfully, False if skill not found
        """
        skill = self._skills.get(name)
        if not skill:
            return False

        skill_file = self._skills_dir / skill.filename / SKILL_FILENAME

        # Parse the new content to validate and extract fields
        frontmatter, body = parse_frontmatter(content)
        if not frontmatter and not body:
            raise ValueError("Invalid skill format: missing frontmatter or content")

        # Write the content
        with open(skill_file, "w") as f:
            f.write(content)

        # Update in-memory skill
        new_name = frontmatter.get("name", name)
        skill.prompt = body.strip()
        skill.description = frontmatter.get("description", "").strip()
        skill.allowed_tools = frontmatter.get("allowed-tools", [])

        # Handle name change
        if new_name != name:
            del self._skills[name]
            skill.name = new_name
            self._skills[new_name] = skill
            if name in self._active_skills:
                self._active_skills.discard(name)
                self._active_skills.add(new_name)

        return True

    def delete_skill(self, name: str) -> bool:
        """Delete a skill.

        Args:
            name: Skill name

        Returns:
            True if deleted, False if not found
        """
        skill = self._skills.get(name)
        if not skill:
            return False

        skill_dir = self._skills_dir / skill.filename
        try:
            # Remove the SKILL.md file
            skill_file = skill_dir / SKILL_FILENAME
            if skill_file.exists():
                skill_file.unlink()
            # Remove the directory if empty
            if skill_dir.exists() and not any(skill_dir.iterdir()):
                skill_dir.rmdir()
        except OSError as e:
            logger.warning(f"Failed to delete skill directory {skill_dir}: {e}")
            return False

        del self._skills[name]
        self._active_skills.discard(name)
        return True

    def get_skill_content(self, name: str) -> Optional[tuple[str, str]]:
        """Get raw markdown content for a skill.

        Args:
            name: Skill name

        Returns:
            Tuple of (content, filepath) or None if not found
        """
        skill = self._skills.get(name)
        if not skill:
            return None

        skill_file = self._skills_dir / skill.filename / SKILL_FILENAME
        try:
            with open(skill_file, "r") as f:
                content = f.read()
            return content, str(skill_file)
        except OSError:
            return None

    @staticmethod
    def draft_skill(name: str, user_description: str, llm) -> tuple[str, str]:
        """Draft a skill using LLM based on user description.

        Args:
            name: Skill name
            user_description: Natural language description of the desired skill
            llm: LLM provider with generate() method

        Returns:
            Tuple of (content, description) - the SKILL.md content and extracted description

        Raises:
            ValueError: If LLM fails to generate valid content
        """
        system_prompt = """You are an expert at creating SKILL files for a data analysis assistant.

Skills are REUSABLE, DOMAIN-SPECIFIC reference materials that can be used standalone or referenced by roles.
Skills contain patterns, SQL queries, metric definitions, and domain knowledge.

A skill file has two parts:

1. **YAML frontmatter** (between ---):
   - name: skill identifier (kebab-case)
   - description: brief description of what domain/patterns this covers
   - allowed-tools: list of tools (typically: list_tables, get_table_schema, run_sql)

2. **Markdown body**: Domain-specific reference content including:
   - Key metrics and their calculations (as tables)
   - Common SQL query patterns (as code blocks)
   - Domain terminology and relationships
   - Best practices for this domain
   - Related skills (e.g., "Related: customer-insights")

Good skills are:
- Narrowly focused on one domain (e.g., "sales-analysis", "customer-retention", "hr-compliance")
- Full of concrete SQL patterns and metric formulas
- Reusable across different roles (e.g., both "Executive" and "Data Analyst" roles can use "sales-analysis" skill)
- Reference material that augments roles with deeper domain knowledge

Output the complete SKILL.md content (frontmatter + markdown body). No explanation outside the skill content."""

        user_prompt = f"""Create a skill named "{name}" based on this description:

{user_description}

Generate a complete SKILL.md file with YAML frontmatter and markdown body containing relevant SQL patterns, metrics, and domain knowledge."""

        result = llm.generate(
            system=system_prompt,
            user_message=user_prompt,
            max_tokens=llm.max_output_tokens,
        )

        content = result.strip()
        # Remove Markdown code block wrapper if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        # Extract description from frontmatter
        description = ""
        if content.startswith("---"):
            try:
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    description = frontmatter.get("description", "")
            except (yaml.YAMLError, AttributeError):
                pass

        return content, description

    @staticmethod
    def skill_from_proof(
        name: str,
        proof_nodes: list[dict],
        proof_summary: str | None,
        original_problem: str,
        llm,
        description: str | None = None,
        script_params: list[dict] | None = None,
        result_schemas: dict[str, list[dict]] | None = None,
    ) -> tuple[str, str]:
        """Distill a completed proof into SKILL.md content.

        Args:
            name: Skill name
            proof_nodes: List of proof node dicts from the proof tree
            proof_summary: LLM-generated narrative summary of the proof
            original_problem: The original problem/claim being proven
            llm: LLM provider for generating skill content
            description: Optional human-readable description
            script_params: List of dicts with 'name' and 'default' keys describing
                          run_proof() keyword arguments.
            result_schemas: Dict mapping dataset name to list of column dicts
                           (each with 'name', 'type', 'nullable' keys).
                           These are the ACTUAL output columns from the script.

        Returns (content, description).
        """
        import json

        system_prompt = """You are an expert at creating SKILL files for a data analysis assistant.

You are converting a completed analysis into a reusable skill. The skill documentation is an INTERFACE DESCRIPTION — it tells a caller what the skill does, what parameters it accepts, and what it returns. It does NOT describe internal implementation details.

A skill file has two parts:

1. **YAML frontmatter** (between ---):
   - name: skill identifier (kebab-case)
   - description: One-line summary of capability — name the domain and data sources.
   - allowed-tools: list of tools (typically: list_tables, get_table_schema, run_sql)

2. **Markdown body** — focused on the EXTERNAL interface:
   - **Capability**: What the skill does (1-2 sentences)
   - **Data Sources**: Which APIs/databases it uses (names and what they provide)
   - **Parameters**: The `run_proof()` function signature — each parameter, its type, default value, and what it controls
   - **Returns**: What datasets are returned (names, key columns, data types)
   - **Usage**: How to call `run_proof()` with examples showing parameter overrides

DO NOT include:
- Internal pipeline stages or derivation chains
- SQL query patterns or join logic
- Intermediate table names or internal variable names
- Implementation details about how data is transformed

Think of this like API documentation: the reader needs to know WHAT it does and HOW TO CALL IT, not how it works inside.

BAD: "The analysis follows a multi-stage derivation chain: 1. selected_breeds (derived) → Random sample..."
GOOD: "Returns a dataset of cat breeds enriched with country, language, and currency data."

BAD: "```sql SELECT eb.*, cd.languages[0] as primary_language FROM enhanced_breeds eb...```"
GOOD: "**breed_limit** (int, default=10): Number of random cat breeds to include."

Output the complete SKILL.md content (frontmatter + markdown body). No explanation outside the skill content."""

        # Serialize proof nodes for the prompt
        nodes_summary = []
        for node in proof_nodes:
            entry = {
                "name": node.get("name", ""),
                "strategy": node.get("strategy", ""),
                "source": node.get("source", ""),
                "formula": node.get("formula", ""),
            }
            nodes_summary.append(entry)

        # Build actual schema documentation
        schema_docs = ""
        if result_schemas:
            schema_lines = ["\n## ACTUAL OUTPUT SCHEMAS (from executed script — use these EXACT column names):\n"]
            for dataset_name, columns in result_schemas.items():
                schema_lines.append(f"### Dataset: `{dataset_name}`")
                for col in columns:
                    col_type = col.get("type", "unknown")
                    schema_lines.append(f"  - `{col['name']}` ({col_type})")
                schema_lines.append("")
            schema_docs = "\n".join(schema_lines)

        user_prompt = f"""Create a skill named "{name}" from this completed analysis.

Original problem: {original_problem}

{f"Summary: {proof_summary}" if proof_summary else ""}

Analysis steps (verified facts with derivation strategies):
{json.dumps(nodes_summary, indent=2)}

Generate a SKILL.md focused on capabilities, parameters, and return values. Do NOT describe internal implementation.

The executable script is `scripts/proof.py` with a `run_proof()` function that returns `dict[str, DataFrame]` (all datasets plus `_result` key for the final output).

{f"run_proof() parameters:{chr(10)}" + chr(10).join(f"- {p['name']}: default={p['default']}" for p in script_params) if script_params else "run_proof() takes no parameters."}
{schema_docs}
CRITICAL: The "Returns" section MUST document the EXACT column names and types shown above. Do NOT invent, rename, or omit any columns. Copy them verbatim from the ACTUAL OUTPUT SCHEMAS."""

        # noinspection DuplicatedCode
        result = llm.generate(
            system=system_prompt,
            user_message=user_prompt,
            max_tokens=llm.max_output_tokens,
        )

        content = result.strip()
        # Remove Markdown code block wrapper if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        # Extract description from frontmatter
        extracted_description = description or ""
        if content.startswith("---"):
            try:
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    extracted_description = frontmatter.get("description", extracted_description)
            except (yaml.YAMLError, AttributeError):
                pass

        return content, extracted_description
