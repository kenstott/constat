# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""User skills for customizing system prompts.

Skills are reusable prompt snippets stored per user in ~/.constat/users/{user_id}/skills/.
Each skill is a YAML file with a name and prompt content.
Users can activate multiple skills at once.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
import logging
import os

logger = logging.getLogger(__name__)

CONSTAT_DIR = Path.home() / ".constat"


def get_skills_dir(user_id: str) -> Path:
    """Get the skills directory for a user."""
    return CONSTAT_DIR / "users" / user_id / "skills"


@dataclass
class Skill:
    """A user-defined skill."""
    name: str
    prompt: str
    description: str = ""
    filename: str = ""


class SkillManager:
    """Manages user skills loaded from ~/.constat/users/{user_id}/skills/."""

    def __init__(self, user_id: str = "default"):
        self._user_id = user_id
        self._skills_dir = get_skills_dir(user_id)
        self._skills: dict[str, Skill] = {}
        self._active_skills: set[str] = set()
        self._ensure_skills_dir()
        self._load_skills()

    def _ensure_skills_dir(self) -> None:
        """Ensure the skills directory exists."""
        self._skills_dir.mkdir(parents=True, exist_ok=True)

    def _load_skills(self) -> None:
        """Load skills from YAML files in the skills directory."""
        self._skills.clear()

        if not self._skills_dir.exists():
            logger.debug(f"No skills directory at {self._skills_dir}")
            return

        for filepath in self._skills_dir.glob("*.yaml"):
            try:
                with open(filepath, "r") as f:
                    data = yaml.safe_load(f) or {}

                name = data.get("name", filepath.stem)
                prompt = data.get("prompt", "").strip()
                description = data.get("description", "").strip()

                if prompt:
                    self._skills[name] = Skill(
                        name=name,
                        prompt=prompt,
                        description=description,
                        filename=filepath.name,
                    )
                    logger.debug(f"Loaded skill: {name} from {filepath.name}")

            except Exception as e:
                logger.warning(f"Failed to load skill from {filepath}: {e}")

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
            name: Skill name (will be used as filename)
            prompt: The skill prompt content
            description: Optional description

        Returns:
            The created Skill object

        Raises:
            ValueError: If skill with this name already exists
        """
        # Sanitize name for filename
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        filename = f"{safe_name}.yaml"
        filepath = self._skills_dir / filename

        if filepath.exists():
            raise ValueError(f"Skill '{name}' already exists")

        skill_data = {
            "name": name,
            "prompt": prompt,
            "description": description,
        }

        with open(filepath, "w") as f:
            yaml.dump(skill_data, f, default_flow_style=False, allow_unicode=True)

        skill = Skill(
            name=name,
            prompt=prompt,
            description=description,
            filename=filename,
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

        filepath = self._skills_dir / skill.filename

        # Load current data
        with open(filepath, "r") as f:
            data = yaml.safe_load(f) or {}

        # Update fields
        if prompt is not None:
            data["prompt"] = prompt
            skill.prompt = prompt
        if description is not None:
            data["description"] = description
            skill.description = description
        if new_name is not None and new_name != name:
            data["name"] = new_name
            # Update in-memory
            del self._skills[name]
            skill.name = new_name
            self._skills[new_name] = skill
            # Update active skills if needed
            if name in self._active_skills:
                self._active_skills.discard(name)
                self._active_skills.add(new_name)

        # Save back
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        return True

    def update_skill_content(self, name: str, content: str) -> bool:
        """Update a skill from raw YAML content.

        Args:
            name: Skill name
            content: Raw YAML content

        Returns:
            True if updated successfully, False if skill not found
        """
        skill = self._skills.get(name)
        if not skill:
            return False

        filepath = self._skills_dir / skill.filename

        # Parse the new content to validate and extract fields
        try:
            data = yaml.safe_load(content) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")

        # Write the content
        with open(filepath, "w") as f:
            f.write(content)

        # Update in-memory skill
        new_name = data.get("name", name)
        skill.prompt = data.get("prompt", "").strip()
        skill.description = data.get("description", "").strip()

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

        filepath = self._skills_dir / skill.filename
        try:
            filepath.unlink()
        except OSError as e:
            logger.warning(f"Failed to delete skill file {filepath}: {e}")
            return False

        del self._skills[name]
        self._active_skills.discard(name)
        return True

    def get_skill_content(self, name: str) -> Optional[tuple[str, str]]:
        """Get raw YAML content for a skill.

        Args:
            name: Skill name

        Returns:
            Tuple of (content, filepath) or None if not found
        """
        skill = self._skills.get(name)
        if not skill:
            return None

        filepath = self._skills_dir / skill.filename
        try:
            with open(filepath, "r") as f:
                content = f.read()
            return (content, str(filepath))
        except OSError:
            return None
