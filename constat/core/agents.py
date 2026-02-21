# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""User agents for customizing system prompts.

Agents are optional. If {base_dir}/agents.yaml exists, users can switch
between defined agents. Each agent adds a prompt to the system prompt.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

AGENTS_FILENAME = "agents.yaml"


def get_agents_file(user_id: str, base_dir: Optional[Path] = None) -> Path:
    """Get the agents file path for a user.

    Args:
        user_id: User identifier
        base_dir: Base .constat directory (defaults to ./.constat)
    """
    if base_dir is None:
        base_dir = Path(".constat")
    return base_dir / user_id / AGENTS_FILENAME


@dataclass
class Agent:
    """A user-defined agent."""
    name: str
    prompt: str
    description: str = ""
    skills: list[str] = field(default_factory=list)


class AgentManager:
    """Manages user agents loaded from {base_dir}/{user_id}/agents.yaml."""

    def __init__(self, user_id: str = "default", base_dir: Optional[Path] = None):
        """Initialize the agent manager.

        Args:
            user_id: User identifier
            base_dir: Base .constat directory. Defaults to ./.constat
        """
        self._user_id = user_id
        self._base_dir = base_dir or Path(".constat")
        self._agents_file = get_agents_file(user_id, self._base_dir)
        self._agents: dict[str, Agent] = {}
        self._active_agent: Optional[str] = None
        self._ensure_agents_dir()
        self._load_agents()

    def _ensure_agents_dir(self) -> None:
        """Ensure the user directory exists."""
        self._agents_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_agents(self) -> None:
        """Load agents from YAML file if it exists."""
        if not self._agents_file.exists():
            logger.debug(f"No agents file at {self._agents_file}")
            return

        try:
            with open(self._agents_file, "r") as f:
                data = yaml.safe_load(f) or {}

            for name, config in data.items():
                if isinstance(config, dict) and "prompt" in config:
                    self._agents[name] = Agent(
                        name=name,
                        prompt=config["prompt"].strip(),
                        description=config.get("description", "").strip(),
                        skills=config.get("skills", []) or [],
                    )
                    logger.debug(f"Loaded agent: {name}")

            logger.info(f"Loaded {len(self._agents)} agents from {self._agents_file}")
        except Exception as e:
            logger.warning(f"Failed to load agents from {self._agents_file}: {e}")

    def reload(self) -> None:
        """Reload agents from file."""
        self._agents.clear()
        self._load_agents()

    def list_agents(self) -> list[str]:
        """Get list of available agent names."""
        return list(self._agents.keys())

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
        """Get the path to the agents file."""
        return self._agents_file

    def get_agent_content(self, name: str) -> Optional[tuple[str, str]]:
        """Get the raw YAML content for a single agent.

        Args:
            name: Agent name

        Returns:
            Tuple of (yaml_content, file_path) or None if agent not found
        """
        if name not in self._agents:
            return None

        agent = self._agents[name]
        # Build YAML for single agent
        agent_data: dict = {
            "prompt": agent.prompt,
            "description": agent.description,
        }
        if agent.skills:
            agent_data["skills"] = agent.skills
        content = yaml.dump({
            name: agent_data,
        }, default_flow_style=False, allow_unicode=True)
        return content, str(self._agents_file)

    def update_agent(self, name: str, prompt: str, description: str = "",
                     skills: list[str] | None = None) -> bool:
        """Update or create an agent.

        Args:
            name: Agent name
            prompt: Agent prompt
            description: Agent description
            skills: Optional list of skill names this agent requires

        Returns:
            True if successful
        """
        # Load current file content to preserve other agents
        if self._agents_file.exists():
            with open(self._agents_file, "r") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        # Update the agent
        agent_data: dict = {
            "prompt": prompt.strip(),
            "description": description.strip(),
        }
        if skills is not None:
            agent_data["skills"] = skills
        elif name in data and isinstance(data[name], dict):
            # Preserve existing skills if not explicitly provided
            existing_skills = data[name].get("skills", [])
            if existing_skills:
                agent_data["skills"] = existing_skills
        data[name] = agent_data

        # Write back
        with open(self._agents_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        # Reload to update internal state
        self.reload()
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

        # Load current file content
        if self._agents_file.exists():
            with open(self._agents_file, "r") as f:
                data = yaml.safe_load(f) or {}
        else:
            return False

        if name not in data:
            return False

        # Remove the agent
        del data[name]

        # Write back
        with open(self._agents_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        # Clear active agent if it was deleted
        if self._active_agent == name:
            self._active_agent = None

        # Reload to update internal state
        self.reload()
        return True

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

        self.update_agent(name, prompt, description, skills=skills)
        return self._agents[name]

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
