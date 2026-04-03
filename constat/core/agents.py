# Copyright (c) 2025 Kenneth Stott
# Canary: df763640-d644-43d1-a7eb-c0c63a393e66
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""User agents for customizing system prompts.

Agents follow the same markdown-with-frontmatter pattern as skills.
Each agent is a directory containing an AGENT.md file.

Directory structure:
    {base_dir}/{user_id}/agents/{agent-name}/
    └── AGENT.md (required)

AGENT.md format:
    ---
    name: agent-name
    description: Brief description
    model: claude-sonnet-4-20250514
    skills:
      - skill-1
      - skill-2
    ---

    The markdown body IS the agent prompt.
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from constat.core.skills import parse_frontmatter

logger = logging.getLogger(__name__)

AGENTS_FILENAME = "agents.yaml"
AGENT_FILENAME = "AGENT.md"


def get_agents_dir(user_id: str, base_dir: Optional[Path] = None) -> Path:
    """Get the agents directory for a user.

    Args:
        user_id: User identifier
        base_dir: Base .constat directory (defaults to ./.constat)
    """
    if base_dir is None:
        base_dir = Path(".constat")
    from constat.core.paths import user_vault_dir
    return user_vault_dir(base_dir, user_id) / "agents"


def get_agents_file(user_id: str, base_dir: Optional[Path] = None) -> Path:
    """Get the legacy agents.yaml file path for a user.

    Args:
        user_id: User identifier
        base_dir: Base .constat directory (defaults to ./.constat)
    """
    if base_dir is None:
        base_dir = Path(".constat")
    from constat.core.paths import user_vault_dir
    return user_vault_dir(base_dir, user_id) / AGENTS_FILENAME


def _build_agent_md(name: str, prompt: str, description: str = "",
                    model: str = "", skills: list[str] | None = None) -> str:
    """Build AGENT.md content from agent fields."""
    frontmatter: dict = {"name": name, "description": description}
    if model:
        frontmatter["model"] = model
    if skills:
        frontmatter["skills"] = skills
    fm_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    return f"---\n{fm_yaml}---\n\n{prompt.strip()}\n"


def _safe_dirname(name: str) -> str:
    """Convert an agent name to a safe directory name."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name.lower())


@dataclass
class Agent:
    """A user-defined agent."""
    name: str
    prompt: str
    description: str = ""
    skills: list[str] = field(default_factory=list)
    domain: str = ""   # owning domain filename ("" = unscoped/global)
    source: str = ""   # "system" | "shared" | "user" | "domain"
    model: str = ""    # model override (e.g., "claude-sonnet-4-20250514")


class AgentManager:
    """Manages agents loaded from AGENT.md files in agents/ directories.

    Precedence order (later overrides earlier by name):
        1. System:  {system_agents_dir}/  (if provided)
        2. Domain:  {domain_dir}/agents/  (for each active domain)
        3. User:    {base_dir}/{user_id}/agents/
    """

    def __init__(self, user_id: str = "default", base_dir: Optional[Path] = None,
                 system_agents_dir: Optional[Path] = None):
        """Initialize the agent manager.

        Args:
            user_id: User identifier
            base_dir: Base .constat directory. Defaults to ./.constat
            system_agents_dir: System agents directory (config_dir/agents/).
        """
        self._user_id = user_id
        self._base_dir = base_dir or Path(".constat")
        self._agents_dir = get_agents_dir(user_id, self._base_dir)
        self._legacy_agents_file = get_agents_file(user_id, self._base_dir)
        self._system_agents_dir = system_agents_dir
        self._domain_agent_dirs: list[tuple[Path, str]] = []  # (agents_dir, domain_filename)
        self._agents: dict[str, Agent] = {}
        self._active_agent: Optional[str] = None
        self._ensure_agents_dir()
        self._migrate_from_yaml()
        self._load_agents()

    def _ensure_agents_dir(self) -> None:
        """Ensure the user agents directory exists."""
        self._agents_dir.mkdir(parents=True, exist_ok=True)

    def _migrate_from_yaml(self) -> None:
        """One-time migration: convert agents.yaml entries to AGENT.md files."""
        yaml_file = self._legacy_agents_file
        if not yaml_file.exists():
            return
        # Only migrate if agents/ dir is empty (no subdirs with AGENT.md)
        has_agent_dirs = any(
            (d / AGENT_FILENAME).exists()
            for d in self._agents_dir.iterdir()
            if d.is_dir()
        ) if self._agents_dir.exists() else False
        if has_agent_dirs:
            return

        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f) or {}

        for name, config in data.items():
            if not isinstance(config, dict) or "prompt" not in config:
                continue
            safe_name = _safe_dirname(name)
            agent_dir = self._agents_dir / safe_name
            agent_dir.mkdir(parents=True, exist_ok=True)
            content = _build_agent_md(
                name=name,
                prompt=config["prompt"].strip(),
                description=config.get("description", "").strip(),
                model=config.get("model", ""),
                skills=config.get("skills", []) or [],
            )
            (agent_dir / AGENT_FILENAME).write_text(content)
            logger.info(f"Migrated agent '{name}' from YAML to {agent_dir / AGENT_FILENAME}")

        # Rename the YAML file to mark it as imported
        imported_path = yaml_file.with_suffix(".yaml.imported")
        yaml_file.rename(imported_path)
        logger.info(f"Renamed {yaml_file} -> {imported_path}")

    def _load_agents_from_dir(self, agents_dir: Path, source: str, domain: str = "") -> None:
        """Load agents from a directory, overriding existing entries by name."""
        if not agents_dir.exists():
            return

        for agent_dir in agents_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            agent_file = agent_dir / AGENT_FILENAME
            if not agent_file.exists():
                continue

            try:
                content = agent_file.read_text()
                frontmatter, body = parse_frontmatter(content)

                name = frontmatter.get("name", agent_dir.name)
                description = frontmatter.get("description", "").strip()
                prompt = body.strip()
                model = frontmatter.get("model", "")
                skills = frontmatter.get("skills", [])

                if prompt:
                    self._agents[name] = Agent(
                        name=name,
                        prompt=prompt,
                        description=description,
                        skills=skills if isinstance(skills, list) else [],
                        model=model,
                        domain=domain,
                        source=source,
                    )
                    logger.debug(f"Loaded agent: {name} from {agent_dir.name}/AGENT.md ({source})")

            except Exception as e:
                logger.warning(f"Failed to load agent from {agent_file}: {e}")

    def _load_agents_from_yaml(self, yaml_file: Path, source: str, domain: str = "") -> None:
        """Load agents from a legacy YAML file (backward compat for domain dirs)."""
        if not yaml_file.exists():
            return
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f) or {}
            for name, config in data.items():
                if isinstance(config, dict) and "prompt" in config:
                    self._agents[name] = Agent(
                        name=name,
                        prompt=config["prompt"].strip(),
                        description=config.get("description", "").strip(),
                        skills=config.get("skills", []) or [],
                        domain=domain,
                        source=source,
                        model=config.get("model", ""),
                    )
            logger.info(f"Loaded agents from legacy YAML: {yaml_file}")
        except Exception as e:
            logger.warning(f"Failed to load agents from {yaml_file}: {e}")

    def _load_agents(self) -> None:
        """Load agents from all directories in precedence order.

        System < domain < user (last wins).
        """
        self._agents.clear()

        # 1. System agents (lowest precedence)
        if self._system_agents_dir:
            self._load_agents_from_dir(self._system_agents_dir, "system")

        # 2. Domain agent dirs
        for agents_dir, domain_filename in self._domain_agent_dirs:
            self._load_agents_from_dir(agents_dir, "domain", domain=domain_filename)

        # 3. User agents (highest precedence)
        self._load_agents_from_dir(self._agents_dir, "user")

        # Assign domain="user" and source="user" to unscoped agents
        for agent in self._agents.values():
            if not agent.domain:
                agent.domain = "user"
            if not agent.source:
                agent.source = "user"

        logger.info(
            f"Loaded {len(self._agents)} agents "
            f"(system={self._system_agents_dir}, "
            f"domains={len(self._domain_agent_dirs)}, "
            f"user={self._agents_dir})"
        )

    def reload(self) -> None:
        """Reload agents from files."""
        self._agents.clear()
        self._load_agents()

    def add_domain_agents(self, domain_dir: Path, domain_filename: str = "") -> None:
        """Load agents from a domain directory and merge into the pool.

        Supports both AGENT.md directories and legacy agents.yaml.

        Args:
            domain_dir: Path to the domain directory (parent of agents/).
            domain_filename: Owning domain filename for scoping.
        """
        agents_subdir = domain_dir / "agents"
        yaml_file = domain_dir / "agents.yaml"

        if agents_subdir.exists():
            entry = (agents_subdir, domain_filename)
            if entry not in self._domain_agent_dirs:
                self._domain_agent_dirs.append(entry)
                self._load_agents()
        elif yaml_file.exists():
            # Backward compat: load from YAML if no agents/ directory
            self._load_agents_from_yaml(yaml_file, "domain", domain=domain_filename)

    def remove_domain_agents(self, domain_filename: str) -> None:
        """Remove all agents belonging to a domain."""
        to_remove = [n for n, a in self._agents.items() if a.domain == domain_filename]
        for name in to_remove:
            del self._agents[name]
            if self._active_agent == name:
                self._active_agent = None
        # Also remove the domain dir entry
        self._domain_agent_dirs = [
            (d, df) for d, df in self._domain_agent_dirs if df != domain_filename
        ]

    def list_agents(self, domain: Optional[str] = None) -> list[str]:
        """Get list of available agent names, optionally filtered by domain."""
        if domain is not None:
            return [n for n, a in self._agents.items() if a.domain == domain]
        return list(self._agents.keys())

    def get_domain_agents(self, domain: str) -> list[Agent]:
        """Get agents belonging to a specific domain."""
        return [a for a in self._agents.values() if a.domain == domain]

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self._agents.get(name)

    def set_active_agent(self, name: Optional[str]) -> bool:
        """Set the active agent.

        Args:
            name: Agent name, or None to clear active agent

        Returns:
            True if agent was set successfully, False if agent not found
        """
        if name is None or name.lower() == "none":
            self._active_agent = None
            return True

        if name not in self._agents:
            return False

        self._active_agent = name
        return True

    @property
    def active_agent(self) -> Optional[Agent]:
        """Get the currently active agent."""
        if self._active_agent:
            return self._agents.get(self._active_agent)
        return None

    @property
    def active_agent_name(self) -> Optional[str]:
        """Get the name of the currently active agent."""
        return self._active_agent

    def get_agent_prompt(self) -> str:
        """Get the prompt for the active agent, or empty string if none."""
        agent = self.active_agent
        return agent.prompt if agent else ""

    @property
    def has_agents(self) -> bool:
        """Check if any agents are defined."""
        return len(self._agents) > 0

    @property
    def agents_file_path(self) -> Path:
        """Get the path to the agents directory.

        Note: Previously returned the agents.yaml path. Now returns the
        agents/ directory. Callers using this for display purposes will
        show the directory path instead.
        """
        return self._agents_dir

    def get_agent_content(self, name: str) -> Optional[tuple[str, str]]:
        """Get the raw AGENT.md content for an agent.

        Args:
            name: Agent name

        Returns:
            Tuple of (content, file_path) or None if agent not found
        """
        agent = self._agents.get(name)
        if not agent:
            return None

        # Find the AGENT.md file
        agent_file = self._find_agent_file(name)
        if not agent_file:
            return None

        content = agent_file.read_text()
        return content, str(agent_file)

    def _find_agent_file(self, name: str) -> Optional[Path]:
        """Find the AGENT.md file for an agent by checking all directories."""
        agent = self._agents.get(name)
        if not agent:
            return None

        # Search in reverse precedence: user > domain > system
        search_dirs = [self._agents_dir]
        for agents_dir, _ in reversed(self._domain_agent_dirs):
            search_dirs.append(agents_dir)
        if self._system_agents_dir:
            search_dirs.append(self._system_agents_dir)

        safe_name = _safe_dirname(name)
        for search_dir in search_dirs:
            # Try safe_name first, then iterate to find by frontmatter name
            candidate = search_dir / safe_name / AGENT_FILENAME
            if candidate.exists():
                return candidate
            # Also check all subdirs in case dirname differs from safe_name
            if search_dir.exists():
                for sub in search_dir.iterdir():
                    if sub.is_dir():
                        f = sub / AGENT_FILENAME
                        if f.exists():
                            try:
                                fm, _ = parse_frontmatter(f.read_text())
                                if fm.get("name") == name:
                                    return f
                            except Exception:
                                pass
        return None

    def create_agent(self, name: str, prompt: str, description: str = "",
                     skills: list[str] | None = None) -> Agent:
        """Create a new agent.

        Args:
            name: Agent name
            prompt: Agent prompt
            description: Agent description
            skills: Optional list of skill names this agent requires

        Returns:
            The created Agent

        Raises:
            ValueError: If agent already exists
        """
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already exists")

        safe_name = _safe_dirname(name)
        agent_dir = self._agents_dir / safe_name
        agent_dir.mkdir(parents=True, exist_ok=True)

        content = _build_agent_md(
            name=name,
            prompt=prompt.strip(),
            description=description.strip() if description else "",
            skills=skills,
        )
        (agent_dir / AGENT_FILENAME).write_text(content)

        agent = Agent(
            name=name,
            prompt=prompt.strip(),
            description=description.strip() if description else "",
            skills=skills or [],
            domain="user",
            source="user",
        )
        self._agents[name] = agent
        return agent

    def update_agent(self, name: str, prompt: str, description: str = "",
                     skills: list[str] | None = None,
                     model: str | None = None) -> bool:
        """Update or create an agent.

        Args:
            name: Agent name
            prompt: Agent prompt
            description: Agent description
            skills: Optional list of skill names this agent requires
            model: Optional model override

        Returns:
            True if successful
        """
        existing = self._agents.get(name)

        # Preserve existing fields if not provided
        if existing:
            if model is None:
                model = existing.model
            if skills is None:
                skills = existing.skills

        safe_name = _safe_dirname(name)
        agent_dir = self._agents_dir / safe_name
        agent_dir.mkdir(parents=True, exist_ok=True)

        content = _build_agent_md(
            name=name,
            prompt=prompt.strip(),
            description=description.strip() if description else "",
            model=model or "",
            skills=skills,
        )
        (agent_dir / AGENT_FILENAME).write_text(content)

        self._agents[name] = Agent(
            name=name,
            prompt=prompt.strip(),
            description=description.strip() if description else "",
            skills=skills or [],
            model=model or "",
            domain="user",
            source="user",
        )
        return True

    def delete_agent(self, name: str) -> bool:
        """Delete an agent.

        Args:
            name: Agent name

        Returns:
            True if deleted, False if not found
        """
        if name not in self._agents:
            return False

        # Find and remove the directory
        safe_name = _safe_dirname(name)
        agent_dir = self._agents_dir / safe_name
        if agent_dir.exists():
            shutil.rmtree(agent_dir)

        # Clear active agent if it was deleted
        if self._active_agent == name:
            self._active_agent = None

        del self._agents[name]
        return True

    @staticmethod
    def draft_agent(name: str, user_description: str, llm,
                    available_skills: list[dict[str, str]] | None = None) -> Agent:
        """Draft an agent using LLM based on user description.

        Args:
            name: Agent name
            user_description: Natural language description of the desired agent
            llm: LLM provider with generate() method
            available_skills: List of {"name": str, "description": str} for skill selection

        Returns:
            The drafted Agent (not yet saved)

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
Available skills that can be attached to agents (select relevant ones):
{skill_lines}

If any skills are relevant to this agent, include them in the "skills" array (use exact names).
If none are relevant, return an empty "skills" array."""

        system_prompt = f"""You are an expert at creating PERSONA agents for a data analysis assistant.

An agent defines a PERSONA - combining communication style, priorities, perspective, and domain context relevant to that agent.

Agents have three components:
1. **description**: A brief (1 sentence) description of the persona
2. **prompt**: Instructions defining the persona's behavior, priorities, and domain context
3. **skills**: List of skill names to activate when this agent is selected

Good agent prompts define:
- Communication style (concise vs detailed, technical vs accessible, formal vs casual)
- Perspective (what matters to this persona - speed, accuracy, risk, cost, compliance, etc.)
- Output preferences (bullet points, executive summaries, detailed breakdowns)
- Domain-specific guidance relevant to this agent's perspective
- What to emphasize or de-emphasize

Examples:
- "Executive" agent: Leads with recommendations, 2-3 bullet max, quantifies impact, skips details
- "HR Analyst" agent: Focus on workforce metrics, compliance awareness, PII sensitivity, pay equity
- "Risk Officer" agent: Highlights uncertainties, flags anomalies, conservative interpretations
{skills_section}
Output ONLY valid JSON with keys: "description", "prompt", and "skills". No explanation."""

        user_prompt = f"""Create an agent named "{name}" based on this description:

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

        return Agent(
            name=name,
            prompt=parsed.get("prompt", ""),
            description=parsed.get("description", ""),
            skills=suggested_skills,
        )
