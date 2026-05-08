# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Skill discovery and loading tools for LLM integration.

This module provides tools for discovering and loading SKILL.md files
that follow the Anthropic skill format. Skills provide reusable
instructions, prompts, and domain knowledge that can be loaded into
context on demand.

Skill File Format (SKILL.md):
    ---
    name: skill-name
    description: What this skill does and when to use it
    allowed-tools:  # Optional: restrict which tools the skill can use
      - Read
      - Bash
    ---

    # Skill Title

    ## Instructions
    Step-by-step guidance for the skill...

    ## Examples
    Concrete examples...

Skill Locations (searched in order):
    1. Project: .constat/skills/
    2. Global: ~/.constat/skills/
    3. Config-specified paths

Link Following:
    Skills can reference additional files via markdown links:
    - Relative links: [indicators](references/indicators.md)
    - URLs: [docs](https://example.com/docs.md)

    Links are discovered when the skill loads but content is fetched
    lazily (on-demand) via get_skill_file or resolve_skill_link.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import yaml


@dataclass
class SkillLink:
    """A link discovered in skill content."""
    text: str  # Link text (e.g., "indicators")
    target: str  # Link target (e.g., "references/indicators.md" or "https://...")
    is_url: bool  # True if target is a URL, False if relative path
    line_number: int  # Line number where link appears

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "target": self.target,
            "is_url": self.is_url,
            "line_number": self.line_number,
        }


@dataclass
class SkillMetadata:
    """Metadata from SKILL.md frontmatter."""
    name: str
    description: str
    allowed_tools: list[str] = field(default_factory=list)
    model: Optional[str] = None
    context: Optional[str] = None  # "fork" for isolated sub-agent
    agent: Optional[str] = None  # Agent type when using context: fork
    user_invocable: bool = True


@dataclass
class Skill:
    """A loaded skill with metadata and content."""
    metadata: SkillMetadata
    content: str  # Full markdown content (including frontmatter stripped)
    path: Path  # Source path for debugging

    # Links discovered in the skill content (parsed lazily)
    links: list[SkillLink] = field(default_factory=list)

    # Cache for loaded files/URLs (lazy loading)
    additional_files: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def description(self) -> str:
        return self.metadata.description

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "allowed_tools": self.metadata.allowed_tools,
            "model": self.metadata.model,
            "user_invocable": self.metadata.user_invocable,
            "path": str(self.path),
            "links": [link.to_dict() for link in self.links],
        }


class SkillManager:
    """
    Manages skill loading and discovery.

    Searches for skills in multiple locations:
    1. Project directory: .constat/skills/
    2. Global directory: ~/.constat/skills/
    3. Config-specified paths (from config.yaml skills.paths)

    Each skill is a directory containing at minimum a SKILL.md file.

    Example config.yaml:
        skills:
          paths:
            - /shared/team-skills/
            - /opt/constat/standard-skills/
    """

    # Default skill search paths
    DEFAULT_PATHS = [
        Path(".constat/skills"),
        Path.home() / ".constat/skills",
    ]

    def __init__(self, additional_paths: Optional[list[Path]] = None):
        """
        Initialize the skill manager.

        Args:
            additional_paths: Additional directories to search for skills
        """
        self.search_paths = list(self.DEFAULT_PATHS)
        if additional_paths:
            self.search_paths.extend(additional_paths)

        # Cache loaded skills
        self._skills: dict[str, Skill] = {}
        self._loaded = False

    @classmethod
    def from_config(cls, config: Any) -> "SkillManager":
        """
        Create a SkillManager from a Config object.

        Reads additional skill paths from config.skills.paths and adds them
        to the default search paths.

        Args:
            config: Config object with skills.paths list

        Returns:
            SkillManager with config paths added

        Example:
            from constat.core.config import Config

            config = Config.from_yaml("config.yaml")
            manager = SkillManager.from_config(config)
        """
        additional_paths = []

        # Get paths from config.skills.paths
        if hasattr(config, "skills") and config.skills:
            if hasattr(config.skills, "paths") and config.skills.paths:
                for path_str in config.skills.paths:
                    # Expand ~ and resolve path
                    path = Path(path_str).expanduser()
                    additional_paths.append(path)

        return cls(additional_paths=additional_paths if additional_paths else None)

    def discover_skills(self) -> list[Skill]:
        """
        Discover all available skills from search paths.

        Returns:
            List of discovered skills with metadata
        """
        if self._loaded:
            return list(self._skills.values())

        for search_path in self.search_paths:
            if not search_path.exists():
                continue

            # Each subdirectory is a potential skill
            for skill_dir in search_path.iterdir():
                if not skill_dir.is_dir():
                    continue

                skill_file = skill_dir / "SKILL.md"
                if not skill_file.exists():
                    continue

                try:
                    skill = self._load_skill_file(skill_file)
                    if skill and skill.name not in self._skills:
                        self._skills[skill.name] = skill
                except Exception:
                    # Skip invalid skill files
                    continue

        self._loaded = True
        return list(self._skills.values())

    def get_skill(self, name: str) -> Optional[Skill]:
        """
        Get a skill by name.

        Args:
            name: The skill name

        Returns:
            The skill if found, None otherwise
        """
        if not self._loaded:
            self.discover_skills()

        return self._skills.get(name)

    def load_skill_content(self, name: str) -> Optional[str]:
        """
        Load the full content of a skill.

        This is called when the LLM wants to add a skill to context.

        Args:
            name: The skill name

        Returns:
            Full skill content as markdown, or None if not found
        """
        skill = self.get_skill(name)
        if not skill:
            return None

        return skill.content

    def load_skill_file(self, name: str, filename: str) -> Optional[str]:
        """
        Load an additional file from a skill directory.

        Skills can reference additional files like reference.md or examples.md.

        Args:
            name: The skill name
            filename: The file to load (e.g., "reference.md")

        Returns:
            File content, or None if not found
        """
        skill = self.get_skill(name)
        if not skill:
            return None

        # Check cache
        if filename in skill.additional_files:
            return skill.additional_files[filename]

        # Try to load from skill directory
        file_path = skill.path.parent / filename
        if file_path.exists() and file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8")
                skill.additional_files[filename] = content
                return content
            except Exception:
                return None

        return None

    def _load_skill_file(self, path: Path) -> Optional[Skill]:
        """
        Load and parse a SKILL.md file.

        Args:
            path: Path to the SKILL.md file

        Returns:
            Parsed Skill object, or None if invalid
        """
        content = path.read_text(encoding="utf-8")

        # Parse frontmatter
        metadata, body = self._parse_frontmatter(content)
        if not metadata:
            return None

        # Validate required fields
        name = metadata.get("name")
        description = metadata.get("description")
        if not name or not description:
            return None

        # Parse allowed-tools (can be comma-separated or list)
        allowed_tools = []
        if "allowed-tools" in metadata:
            tools_value = metadata["allowed-tools"]
            if isinstance(tools_value, str):
                allowed_tools = [t.strip() for t in tools_value.split(",")]
            elif isinstance(tools_value, list):
                allowed_tools = tools_value

        skill_metadata = SkillMetadata(
            name=name,
            description=description,
            allowed_tools=allowed_tools,
            model=metadata.get("model"),
            context=metadata.get("context"),
            agent=metadata.get("agent"),
            user_invocable=metadata.get("user-invocable", True),
        )

        # Parse links from content (lazy discovery - content not fetched yet)
        links = self._parse_links(body)

        return Skill(
            metadata=skill_metadata,
            content=body,
            path=path,
            links=links,
        )

    def _parse_links(self, content: str) -> list[SkillLink]:
        """
        Parse markdown links from content.

        Extracts both relative file links and URLs for lazy loading.
        Excludes anchor-only links (#section) and image links.

        Args:
            content: Markdown content to parse

        Returns:
            List of discovered links
        """
        links = []
        # Match markdown links: [text](target)
        # Excludes images: ![alt](src)
        link_pattern = re.compile(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)')

        for line_num, line in enumerate(content.split('\n'), start=1):
            for match in link_pattern.finditer(line):
                text = match.group(1)
                target = match.group(2)

                # Skip anchor-only links
                if target.startswith('#'):
                    continue

                # Skip mailto: and other non-file links
                if target.startswith('mailto:') or target.startswith('javascript:'):
                    continue

                # Determine if this is a URL or relative path
                parsed = urlparse(target)
                is_url = bool(parsed.scheme and parsed.netloc)

                links.append(SkillLink(
                    text=text,
                    target=target,
                    is_url=is_url,
                    line_number=line_num,
                ))

        return links

    def list_skill_links(self, name: str) -> list[SkillLink]:
        """
        List all links discovered in a skill's content.

        This enables the LLM to see what references are available
        before deciding which ones to load.

        Args:
            name: The skill name

        Returns:
            List of discovered links, or empty list if skill not found
        """
        skill = self.get_skill(name)
        if not skill:
            return []

        return skill.links

    def resolve_skill_link(self, name: str, target: str) -> Optional[str]:
        """
        Resolve and fetch a link's content.

        For relative paths: loads from skill directory.
        For URLs: fetches via HTTP (with caching).

        Args:
            name: The skill name
            target: The link target (path or URL)

        Returns:
            Content of the linked resource, or None if not found/failed
        """
        skill = self.get_skill(name)
        if not skill:
            return None

        # Check cache first
        if target in skill.additional_files:
            return skill.additional_files[target]

        # Determine if URL or relative path
        parsed = urlparse(target)
        is_url = bool(parsed.scheme and parsed.netloc)

        if is_url:
            # Fetch URL
            content = self._fetch_url(target)
        else:
            # Load relative to skill directory
            file_path = skill.path.parent / target
            if file_path.exists() and file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8")
                except Exception:
                    content = None
            else:
                content = None

        # Cache the result
        if content is not None:
            skill.additional_files[target] = content

        return content

    def _fetch_url(self, url: str, timeout: int = 30) -> Optional[str]:
        """
        Fetch content from a URL.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            Response text, or None if failed
        """
        try:
            import httpx
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.text
        except ImportError:
            # Fall back to urllib if httpx not available
            try:
                import urllib.request
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    return response.read().decode("utf-8")
            except Exception:
                return None
        except Exception:
            return None

    def _parse_frontmatter(self, content: str) -> tuple[Optional[dict], str]:
        """
        Parse YAML frontmatter from markdown content.

        Args:
            content: Full markdown content with frontmatter

        Returns:
            Tuple of (metadata dict, body content) or (None, content) if no frontmatter
        """
        # Check for frontmatter delimiter
        if not content.startswith("---"):
            return None, content

        # Find the closing delimiter
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not match:
            return None, content

        frontmatter_str = match.group(1)
        body = match.group(2)

        try:
            metadata = yaml.safe_load(frontmatter_str)
            return metadata, body
        except yaml.YAMLError:
            return None, content


class SkillDiscoveryTools:
    """
    Tool interface for skill discovery and loading.

    Provides two main tools:
    - list_skills: Returns available skills with descriptions
    - load_skill: Loads a skill's content into context
    """

    def __init__(self, skill_manager: Optional[SkillManager] = None):
        """
        Initialize skill discovery tools.

        Args:
            skill_manager: SkillManager instance, or None to create default
        """
        self.manager = skill_manager or SkillManager()

    def list_skills(self) -> list[dict[str, Any]]:
        """
        List all available skills with their metadata.

        Returns:
            List of skill info dicts with name, description, allowed_tools
        """
        skills = self.manager.discover_skills()

        return [
            {
                "name": skill.name,
                "description": skill.description,
                "allowed_tools": skill.metadata.allowed_tools,
                "user_invocable": skill.metadata.user_invocable,
            }
            for skill in skills
            if skill.metadata.user_invocable
        ]

    def load_skill(self, name: str) -> dict[str, Any]:
        """
        Load a skill's content to add to context.

        Args:
            name: The skill name to load

        Returns:
            Dict with skill content and metadata, or error if not found
        """
        skill = self.manager.get_skill(name)
        if not skill:
            return {
                "error": f"Skill not found: {name}",
                "available_skills": [s.name for s in self.manager.discover_skills()],
            }

        # Include discovered links for lazy loading
        links_info = [
            {"text": link.text, "target": link.target, "is_url": link.is_url}
            for link in skill.links
        ]

        return {
            "name": skill.name,
            "description": skill.description,
            "content": skill.content,
            "allowed_tools": skill.metadata.allowed_tools,
            "links": links_info,
        }

    def list_skill_links(self, name: str) -> dict[str, Any]:
        """
        List all links discovered in a skill's content.

        Use this to see what references are available before loading them.
        Links are parsed when the skill loads but content is not fetched
        until you call resolve_skill_link.

        Args:
            name: The skill name

        Returns:
            Dict with list of discovered links, or error if skill not found
        """
        skill = self.manager.get_skill(name)
        if not skill:
            return {
                "error": f"Skill not found: {name}",
                "available_skills": [s.name for s in self.manager.discover_skills()],
            }

        return {
            "skill": name,
            "links": [
                {
                    "text": link.text,
                    "target": link.target,
                    "is_url": link.is_url,
                    "line_number": link.line_number,
                }
                for link in skill.links
            ],
        }

    def resolve_skill_link(self, name: str, target: str) -> dict[str, Any]:
        """
        Resolve and fetch a link's content (lazy loading).

        For relative paths: loads from skill directory.
        For URLs: fetches via HTTP.

        Results are cached so subsequent calls for the same target
        return immediately.

        Args:
            name: The skill name
            target: The link target (relative path or URL)

        Returns:
            Dict with content, or error if not found/failed
        """
        content = self.manager.resolve_skill_link(name, target)
        if content is None:
            return {
                "error": f"Failed to resolve link: {target} in skill {name}",
                "hint": "Check that the file exists or URL is accessible",
            }

        return {
            "skill": name,
            "target": target,
            "content": content,
        }

    def get_skill_file(self, name: str, filename: str) -> dict[str, Any]:
        """
        Load an additional file from a skill directory.

        Skills can reference files like reference.md or examples.md for
        progressive disclosure of detailed information.

        Args:
            name: The skill name
            filename: The file to load (e.g., "reference.md")

        Returns:
            Dict with file content, or error if not found
        """
        content = self.manager.load_skill_file(name, filename)
        if content is None:
            return {
                "error": f"File not found: {filename} in skill {name}",
            }

        return {
            "skill": name,
            "filename": filename,
            "content": content,
        }


# Tool schemas for LLM integration
SKILL_TOOL_SCHEMAS = [
    {
        "name": "list_skills",
        "description": (
            "List all available skills. Skills are reusable instructions and "
            "domain knowledge that can be loaded into context. Use this to "
            "discover what skills are available before loading one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "load_skill",
        "description": (
            "Load a skill's content into context. Use this when you need "
            "specialized instructions or domain knowledge for a task. "
            "The skill's content will be added to help guide your response."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the skill to load",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_skill_file",
        "description": (
            "Load an additional file from a skill directory. Skills can have "
            "supplementary files like reference.md or examples.md that contain "
            "detailed information. Use this for progressive disclosure of "
            "detailed content when needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name",
                },
                "filename": {
                    "type": "string",
                    "description": "The file to load (e.g., 'reference.md')",
                },
            },
            "required": ["name", "filename"],
        },
    },
    {
        "name": "list_skill_links",
        "description": (
            "List all links discovered in a skill's content. Links are parsed "
            "when the skill loads but content is NOT fetched until you call "
            "resolve_skill_link. Use this to see what references are available "
            "before deciding which ones to load."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "resolve_skill_link",
        "description": (
            "Resolve and fetch a link's content (lazy loading). For relative "
            "paths, loads from the skill directory. For URLs, fetches via HTTP. "
            "Results are cached so subsequent calls return immediately. Use this "
            "to load referenced content when needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name",
                },
                "target": {
                    "type": "string",
                    "description": "The link target (relative path or URL)",
                },
            },
            "required": ["name", "target"],
        },
    },
]
