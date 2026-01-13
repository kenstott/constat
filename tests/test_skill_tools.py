"""Tests for skill discovery and loading tools."""

import tempfile
from pathlib import Path

import pytest

from constat.discovery.skill_tools import (
    Skill,
    SkillDiscoveryTools,
    SkillManager,
    SkillMetadata,
    SKILL_TOOL_SCHEMAS,
)


class TestSkillManager:
    """Tests for SkillManager."""

    def test_parse_frontmatter_valid(self):
        """Test parsing valid YAML frontmatter."""
        manager = SkillManager()

        content = """---
name: test-skill
description: A test skill for testing
allowed-tools:
  - Read
  - Bash
---

# Test Skill

This is the skill content.
"""
        metadata, body = manager._parse_frontmatter(content)

        assert metadata is not None
        assert metadata["name"] == "test-skill"
        assert metadata["description"] == "A test skill for testing"
        assert metadata["allowed-tools"] == ["Read", "Bash"]
        assert "# Test Skill" in body
        assert "This is the skill content." in body

    def test_parse_frontmatter_comma_separated_tools(self):
        """Test parsing frontmatter with comma-separated tools."""
        manager = SkillManager()

        content = """---
name: my-skill
description: A skill
allowed-tools: Read, Write, Bash
---

Content here.
"""
        metadata, body = manager._parse_frontmatter(content)

        assert metadata is not None
        assert metadata["allowed-tools"] == "Read, Write, Bash"

    def test_parse_frontmatter_no_frontmatter(self):
        """Test parsing content without frontmatter."""
        manager = SkillManager()

        content = "# Just a markdown file\n\nNo frontmatter here."
        metadata, body = manager._parse_frontmatter(content)

        assert metadata is None
        assert body == content

    def test_parse_frontmatter_invalid_yaml(self):
        """Test parsing invalid YAML frontmatter."""
        manager = SkillManager()

        content = """---
name: [invalid: yaml
description: broken
---

Content.
"""
        metadata, body = manager._parse_frontmatter(content)

        # Invalid YAML should return None for metadata
        assert metadata is None

    def test_load_skill_file(self, tmp_path):
        """Test loading a skill from a SKILL.md file."""
        # Create skill directory and file
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        skill_content = """---
name: test-skill
description: A test skill for unit testing
allowed-tools:
  - Read
  - Grep
model: claude-sonnet
user-invocable: true
---

# Test Skill

## Instructions
1. Do this
2. Then that

## Examples
Here are examples...
"""
        (skill_dir / "SKILL.md").write_text(skill_content)

        manager = SkillManager(additional_paths=[tmp_path])
        skills = manager.discover_skills()

        assert len(skills) == 1
        skill = skills[0]
        assert skill.name == "test-skill"
        assert skill.description == "A test skill for unit testing"
        assert skill.metadata.allowed_tools == ["Read", "Grep"]
        assert skill.metadata.model == "claude-sonnet"
        assert skill.metadata.user_invocable is True
        assert "# Test Skill" in skill.content

    def test_discover_skills_multiple(self, tmp_path):
        """Test discovering multiple skills."""
        # Create two skill directories
        for i, name in enumerate(["skill-a", "skill-b"]):
            skill_dir = tmp_path / name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: Skill {i+1} description
---

# Skill {name}
Content for skill {name}.
""")

        manager = SkillManager(additional_paths=[tmp_path])
        skills = manager.discover_skills()

        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"skill-a", "skill-b"}

    def test_discover_skills_skips_invalid(self, tmp_path):
        """Test that invalid skills are skipped."""
        # Valid skill
        valid_dir = tmp_path / "valid-skill"
        valid_dir.mkdir()
        (valid_dir / "SKILL.md").write_text("""---
name: valid-skill
description: A valid skill
---
Content.
""")

        # Invalid skill (missing required fields)
        invalid_dir = tmp_path / "invalid-skill"
        invalid_dir.mkdir()
        (invalid_dir / "SKILL.md").write_text("""---
name: invalid-skill
---
No description field!
""")

        # Directory without SKILL.md
        no_skill_dir = tmp_path / "no-skill"
        no_skill_dir.mkdir()

        manager = SkillManager(additional_paths=[tmp_path])
        skills = manager.discover_skills()

        assert len(skills) == 1
        assert skills[0].name == "valid-skill"

    def test_get_skill(self, tmp_path):
        """Test getting a skill by name."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: my-skill
description: My skill
---
Content.
""")

        manager = SkillManager(additional_paths=[tmp_path])

        skill = manager.get_skill("my-skill")
        assert skill is not None
        assert skill.name == "my-skill"

        # Non-existent skill
        assert manager.get_skill("nonexistent") is None

    def test_load_skill_content(self, tmp_path):
        """Test loading skill content."""
        skill_dir = tmp_path / "content-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: content-skill
description: Skill with content
---

# Main Instructions

Do these things.
""")

        manager = SkillManager(additional_paths=[tmp_path])

        content = manager.load_skill_content("content-skill")
        assert content is not None
        assert "# Main Instructions" in content
        assert "Do these things." in content

        # Non-existent skill
        assert manager.load_skill_content("nonexistent") is None

    def test_load_skill_file_additional(self, tmp_path):
        """Test loading additional files from a skill directory."""
        skill_dir = tmp_path / "multi-file-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: multi-file-skill
description: Skill with multiple files
---
Main content.
""")

        (skill_dir / "reference.md").write_text("""
# API Reference
Detailed API documentation here.
""")

        (skill_dir / "examples.md").write_text("""
# Examples
Example 1: ...
Example 2: ...
""")

        manager = SkillManager(additional_paths=[tmp_path])

        # Load main skill first
        manager.get_skill("multi-file-skill")

        # Load additional files
        reference = manager.load_skill_file("multi-file-skill", "reference.md")
        assert reference is not None
        assert "API Reference" in reference

        examples = manager.load_skill_file("multi-file-skill", "examples.md")
        assert examples is not None
        assert "Example 1" in examples

        # Non-existent file
        assert manager.load_skill_file("multi-file-skill", "nonexistent.md") is None

    def test_default_paths(self):
        """Test that default paths are set correctly."""
        manager = SkillManager()

        assert Path(".constat/skills") in manager.search_paths
        assert Path.home() / ".constat/skills" in manager.search_paths

    def test_additional_paths(self, tmp_path):
        """Test adding additional search paths."""
        extra_path = tmp_path / "extra"
        extra_path.mkdir()

        manager = SkillManager(additional_paths=[extra_path])

        assert extra_path in manager.search_paths
        # Default paths should still be there
        assert Path(".constat/skills") in manager.search_paths


class TestSkillDiscoveryTools:
    """Tests for SkillDiscoveryTools."""

    def test_list_skills(self, tmp_path):
        """Test listing skills via tool interface."""
        skill_dir = tmp_path / "list-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: list-skill
description: A skill for listing test
allowed-tools:
  - Read
---
Content.
""")

        manager = SkillManager(additional_paths=[tmp_path])
        tools = SkillDiscoveryTools(manager)

        skills = tools.list_skills()

        assert len(skills) == 1
        assert skills[0]["name"] == "list-skill"
        assert skills[0]["description"] == "A skill for listing test"
        assert skills[0]["allowed_tools"] == ["Read"]

    def test_list_skills_excludes_non_invocable(self, tmp_path):
        """Test that non-user-invocable skills are excluded from list."""
        skill_dir = tmp_path / "hidden-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: hidden-skill
description: A hidden skill
user-invocable: false
---
Content.
""")

        manager = SkillManager(additional_paths=[tmp_path])
        tools = SkillDiscoveryTools(manager)

        skills = tools.list_skills()
        assert len(skills) == 0

    def test_load_skill(self, tmp_path):
        """Test loading a skill via tool interface."""
        skill_dir = tmp_path / "load-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: load-skill
description: A skill to load
allowed-tools:
  - Read
  - Write
---

# Instructions

Step 1: Do this
Step 2: Do that
""")

        manager = SkillManager(additional_paths=[tmp_path])
        tools = SkillDiscoveryTools(manager)

        result = tools.load_skill("load-skill")

        assert "error" not in result
        assert result["name"] == "load-skill"
        assert result["description"] == "A skill to load"
        assert "# Instructions" in result["content"]
        assert result["allowed_tools"] == ["Read", "Write"]

    def test_load_skill_not_found(self, tmp_path):
        """Test loading a non-existent skill."""
        manager = SkillManager(additional_paths=[tmp_path])
        tools = SkillDiscoveryTools(manager)

        result = tools.load_skill("nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()
        assert "available_skills" in result

    def test_get_skill_file(self, tmp_path):
        """Test getting an additional skill file."""
        skill_dir = tmp_path / "file-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: file-skill
description: A skill with files
---
Main.
""")
        (skill_dir / "extra.md").write_text("Extra content here.")

        manager = SkillManager(additional_paths=[tmp_path])
        tools = SkillDiscoveryTools(manager)

        result = tools.get_skill_file("file-skill", "extra.md")

        assert "error" not in result
        assert result["skill"] == "file-skill"
        assert result["filename"] == "extra.md"
        assert "Extra content here." in result["content"]

    def test_get_skill_file_not_found(self, tmp_path):
        """Test getting a non-existent skill file."""
        skill_dir = tmp_path / "no-file-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: no-file-skill
description: A skill without extra files
---
Main.
""")

        manager = SkillManager(additional_paths=[tmp_path])
        tools = SkillDiscoveryTools(manager)

        result = tools.get_skill_file("no-file-skill", "missing.md")

        assert "error" in result


class TestSkillToolSchemas:
    """Tests for skill tool schemas."""

    def test_schemas_exist(self):
        """Test that skill tool schemas are defined."""
        assert len(SKILL_TOOL_SCHEMAS) == 3

        names = {s["name"] for s in SKILL_TOOL_SCHEMAS}
        assert names == {"list_skills", "load_skill", "get_skill_file"}

    def test_list_skills_schema(self):
        """Test list_skills schema structure."""
        schema = next(s for s in SKILL_TOOL_SCHEMAS if s["name"] == "list_skills")

        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert schema["input_schema"]["required"] == []

    def test_load_skill_schema(self):
        """Test load_skill schema structure."""
        schema = next(s for s in SKILL_TOOL_SCHEMAS if s["name"] == "load_skill")

        assert "description" in schema
        assert "input_schema" in schema
        assert "name" in schema["input_schema"]["properties"]
        assert "name" in schema["input_schema"]["required"]

    def test_get_skill_file_schema(self):
        """Test get_skill_file schema structure."""
        schema = next(s for s in SKILL_TOOL_SCHEMAS if s["name"] == "get_skill_file")

        assert "description" in schema
        assert "input_schema" in schema
        assert "name" in schema["input_schema"]["properties"]
        assert "filename" in schema["input_schema"]["properties"]
        assert set(schema["input_schema"]["required"]) == {"name", "filename"}


class TestSkillIntegration:
    """Integration tests for skill system."""

    def test_skill_to_dict(self, tmp_path):
        """Test skill serialization."""
        skill_dir = tmp_path / "dict-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: dict-skill
description: A serializable skill
allowed-tools:
  - Read
model: claude-sonnet
---
Content.
""")

        manager = SkillManager(additional_paths=[tmp_path])
        skill = manager.get_skill("dict-skill")

        d = skill.to_dict()
        assert d["name"] == "dict-skill"
        assert d["description"] == "A serializable skill"
        assert d["allowed_tools"] == ["Read"]
        assert d["model"] == "claude-sonnet"
        assert "path" in d
