# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""User roles for customizing system prompts.

Roles are optional. If {base_dir}/roles.yaml exists, users can switch
between defined roles. Each role adds a prompt to the system prompt.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

ROLES_FILENAME = "roles.yaml"


def get_roles_file(user_id: str, base_dir: Optional[Path] = None) -> Path:
    """Get the roles file path for a user.

    Args:
        user_id: User identifier
        base_dir: Base .constat directory (defaults to ./.constat)
    """
    if base_dir is None:
        base_dir = Path(".constat")
    return base_dir / user_id / ROLES_FILENAME


@dataclass
class Role:
    """A user-defined role."""
    name: str
    prompt: str
    description: str = ""
    skills: list[str] = field(default_factory=list)


class RoleManager:
    """Manages user roles loaded from {base_dir}/{user_id}/roles.yaml."""

    def __init__(self, user_id: str = "default", base_dir: Optional[Path] = None):
        """Initialize the role manager.

        Args:
            user_id: User identifier
            base_dir: Base .constat directory. Defaults to ./.constat
        """
        self._user_id = user_id
        self._base_dir = base_dir or Path(".constat")
        self._roles_file = get_roles_file(user_id, self._base_dir)
        self._roles: dict[str, Role] = {}
        self._active_role: Optional[str] = None
        self._ensure_roles_dir()
        self._load_roles()

    def _ensure_roles_dir(self) -> None:
        """Ensure the user directory exists."""
        self._roles_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_roles(self) -> None:
        """Load roles from YAML file if it exists."""
        if not self._roles_file.exists():
            logger.debug(f"No roles file at {self._roles_file}")
            return

        try:
            with open(self._roles_file, "r") as f:
                data = yaml.safe_load(f) or {}

            for name, config in data.items():
                if isinstance(config, dict) and "prompt" in config:
                    self._roles[name] = Role(
                        name=name,
                        prompt=config["prompt"].strip(),
                        description=config.get("description", "").strip(),
                        skills=config.get("skills", []) or [],
                    )
                    logger.debug(f"Loaded role: {name}")

            logger.info(f"Loaded {len(self._roles)} roles from {self._roles_file}")
        except Exception as e:
            logger.warning(f"Failed to load roles from {self._roles_file}: {e}")

    def reload(self) -> None:
        """Reload roles from file."""
        self._roles.clear()
        self._load_roles()

    def list_roles(self) -> list[str]:
        """Get list of available role names."""
        return list(self._roles.keys())

    def get_role(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        return self._roles.get(name)

    def set_active_role(self, name: Optional[str]) -> bool:
        """Set the active role.

        Args:
            name: Role name, or None to clear active role

        Returns:
            True if role was set successfully, False if role not found
        """
        if name is None or name.lower() == "none":
            self._active_role = None
            return True

        if name not in self._roles:
            return False

        self._active_role = name
        return True

    @property
    def active_role(self) -> Optional[Role]:
        """Get the currently active role."""
        if self._active_role:
            return self._roles.get(self._active_role)
        return None

    @property
    def active_role_name(self) -> Optional[str]:
        """Get the name of the currently active role."""
        return self._active_role

    def get_role_prompt(self) -> str:
        """Get the prompt for the active role, or empty string if none."""
        role = self.active_role
        return role.prompt if role else ""

    @property
    def has_roles(self) -> bool:
        """Check if any roles are defined."""
        return len(self._roles) > 0

    @property
    def roles_file_path(self) -> Path:
        """Get the path to the roles file."""
        return self._roles_file

    def get_role_content(self, name: str) -> Optional[tuple[str, str]]:
        """Get the raw YAML content for a single role.

        Args:
            name: Role name

        Returns:
            Tuple of (yaml_content, file_path) or None if role not found
        """
        if name not in self._roles:
            return None

        role = self._roles[name]
        # Build YAML for single role
        role_data: dict = {
            "prompt": role.prompt,
            "description": role.description,
        }
        if role.skills:
            role_data["skills"] = role.skills
        content = yaml.dump({
            name: role_data,
        }, default_flow_style=False, allow_unicode=True)
        return content, str(self._roles_file)

    def update_role(self, name: str, prompt: str, description: str = "",
                    skills: list[str] | None = None) -> bool:
        """Update or create a role.

        Args:
            name: Role name
            prompt: Role prompt
            description: Role description
            skills: Optional list of skill names this role requires

        Returns:
            True if successful
        """
        # Load current file content to preserve other roles
        if self._roles_file.exists():
            with open(self._roles_file, "r") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        # Update the role
        role_data: dict = {
            "prompt": prompt.strip(),
            "description": description.strip(),
        }
        if skills is not None:
            role_data["skills"] = skills
        elif name in data and isinstance(data[name], dict):
            # Preserve existing skills if not explicitly provided
            existing_skills = data[name].get("skills", [])
            if existing_skills:
                role_data["skills"] = existing_skills
        data[name] = role_data

        # Write back
        with open(self._roles_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        # Reload to update internal state
        self.reload()
        return True

    def delete_role(self, name: str) -> bool:
        """Delete a role.

        Args:
            name: Role name

        Returns:
            True if deleted, False if not found
        """
        if name not in self._roles:
            return False

        # Load current file content
        if self._roles_file.exists():
            with open(self._roles_file, "r") as f:
                data = yaml.safe_load(f) or {}
        else:
            return False

        if name not in data:
            return False

        # Remove the role
        del data[name]

        # Write back
        with open(self._roles_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        # Clear active role if it was deleted
        if self._active_role == name:
            self._active_role = None

        # Reload to update internal state
        self.reload()
        return True

    def create_role(self, name: str, prompt: str, description: str = "",
                    skills: list[str] | None = None) -> Role:
        """Create a new role.

        Args:
            name: Role name
            prompt: Role prompt
            description: Role description
            skills: Optional list of skill names this role requires

        Returns:
            The created Role

        Raises:
            ValueError: If role already exists
        """
        if name in self._roles:
            raise ValueError(f"Role '{name}' already exists")

        self.update_role(name, prompt, description, skills=skills)
        return self._roles[name]

    @staticmethod
    def draft_role(name: str, user_description: str, llm,
                   available_skills: list[dict[str, str]] | None = None) -> Role:
        """Draft a role using LLM based on user description.

        Args:
            name: Role name
            user_description: Natural language description of the desired role
            llm: LLM provider with generate() method
            available_skills: List of {"name": str, "description": str} for skill selection

        Returns:
            The drafted Role (not yet saved)

        Raises:
            ValueError: If LLM fails to generate valid content
        """
        import json

        skills_section = ""
        if available_skills:
            skill_lines = "\n".join(
                f"- **{s['name']}**: {s['description']}" for s in available_skills
            )
            skills_section = f"""
Available skills that can be attached to roles (select relevant ones):
{skill_lines}

If any skills are relevant to this role, include them in the "skills" array (use exact names).
If none are relevant, return an empty "skills" array."""

        system_prompt = f"""You are an expert at creating PERSONA roles for a data analysis assistant.

A role defines a PERSONA - combining communication style, priorities, perspective, and domain context relevant to that role.

Roles have three components:
1. **description**: A brief (1 sentence) description of the persona
2. **prompt**: Instructions defining the persona's behavior, priorities, and domain context
3. **skills**: List of skill names to activate when this role is selected

Good role prompts define:
- Communication style (concise vs detailed, technical vs accessible, formal vs casual)
- Perspective (what matters to this persona - speed, accuracy, risk, cost, compliance, etc.)
- Output preferences (bullet points, executive summaries, detailed breakdowns)
- Domain-specific guidance relevant to this role's perspective
- What to emphasize or de-emphasize

Examples:
- "Executive" role: Leads with recommendations, 2-3 bullet max, quantifies impact, skips details
- "HR Analyst" role: Focus on workforce metrics, compliance awareness, PII sensitivity, pay equity
- "Risk Officer" role: Highlights uncertainties, flags anomalies, conservative interpretations
{skills_section}
Output ONLY valid JSON with keys: "description", "prompt", and "skills". No explanation."""

        user_prompt = f"""Create a role named "{name}" based on this description:

{user_description}

Return JSON with "description" (brief summary), "prompt" (detailed instructions), and "skills" (list of skill names)."""

        result = llm.generate(
            system=system_prompt,
            user_message=user_prompt,
            max_tokens=llm.max_output_tokens,
        )

        # Parse the JSON response
        content = result.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

        # Validate skills against available list
        suggested_skills = parsed.get("skills", [])
        if available_skills:
            valid_names = {s["name"] for s in available_skills}
            suggested_skills = [s for s in suggested_skills if s in valid_names]

        return Role(
            name=name,
            prompt=parsed.get("prompt", ""),
            description=parsed.get("description", ""),
            skills=suggested_skills,
        )
