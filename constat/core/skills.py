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


class SkillManager:
    """Manages user skills loaded from {base_dir}/{user_id}/skills/."""

    def __init__(self, user_id: str = "default", base_dir: Optional[Path] = None):
        """Initialize the skill manager.

        Args:
            user_id: User identifier
            base_dir: Base .constat directory. Defaults to ./.constat
        """
        self._user_id = user_id
        self._base_dir = base_dir or Path(".constat")
        self._skills_dir = get_skills_dir(user_id, self._base_dir)
        self._skills: dict[str, Skill] = {}
        self._active_skills: set[str] = set()
        self._ensure_skills_dir()
        self._load_skills()

    def _ensure_skills_dir(self) -> None:
        """Ensure the skills directory exists."""
        self._skills_dir.mkdir(parents=True, exist_ok=True)

    def _load_skills(self) -> None:
        """Load skills from SKILL.md files in skill directories."""
        self._skills.clear()

        if not self._skills_dir.exists():
            logger.debug(f"No skills directory at {self._skills_dir}")
            return

        # Look for directories containing SKILL.md
        for skill_dir in self._skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / SKILL_FILENAME
            if not skill_file.exists():
                continue

            try:
                with open(skill_file, "r") as f:
                    content = f.read()

                frontmatter, body = parse_frontmatter(content)

                # Required/recommended fields
                name = frontmatter.get("name", skill_dir.name)
                description = frontmatter.get("description", "").strip()
                prompt = body.strip()

                # Tool restrictions
                allowed_tools = frontmatter.get("allowed-tools", [])

                # Invocation control
                disable_model_invocation = frontmatter.get("disable-model-invocation", False)
                user_invocable = frontmatter.get("user-invocable", True)

                # Execution context
                context = frontmatter.get("context", "")
                agent = frontmatter.get("agent", "")
                model = frontmatter.get("model", "")

                # UI hints
                argument_hint = frontmatter.get("argument-hint", "")

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
                    )
                    logger.debug(f"Loaded skill: {name} from {skill_dir.name}/SKILL.md")

            except Exception as e:
                logger.warning(f"Failed to load skill from {skill_file}: {e}")

        logger.info(f"Loaded {len(self._skills)} skills from {self._skills_dir}")

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
        """Get the path to the skills directory."""
        return self._skills_dir

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
            return (content, str(skill_file))
        except OSError:
            return None

    def draft_skill(self, name: str, user_description: str, llm) -> tuple[str, str]:
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
            max_tokens=self.llm.max_output_tokens,
        )

        content = result.strip()
        # Remove markdown code block wrapper if present
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
            except Exception:
                pass

        return content, description
