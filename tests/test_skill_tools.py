# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for skill discovery and loading tools."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from constat.discovery.skill_tools import (
    Skill,
    SkillDiscoveryTools,
    SkillLink,
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

        # Filter to only test skill (default paths may have other skills)
        test_skills = [s for s in skills if s.name == "test-skill"]
        assert len(test_skills) == 1
        skill = test_skills[0]
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

        # Filter to only test skills (default paths may have other skills)
        test_skills = [s for s in skills if s.name in {"skill-a", "skill-b"}]
        assert len(test_skills) == 2
        names = {s.name for s in test_skills}
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

        # Filter to only test skills (default paths may have other skills)
        test_skills = [s for s in skills if s.name in {"valid-skill", "invalid-skill"}]
        assert len(test_skills) == 1
        assert test_skills[0].name == "valid-skill"

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

        # Filter to only the test skill (default paths may have other skills)
        test_skills = [s for s in skills if s["name"] == "list-skill"]
        assert len(test_skills) == 1
        assert test_skills[0]["name"] == "list-skill"
        assert test_skills[0]["description"] == "A skill for listing test"
        assert test_skills[0]["allowed_tools"] == ["Read"]

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

        # The hidden skill should not be in the list
        hidden_skills = [s for s in skills if s["name"] == "hidden-skill"]
        assert len(hidden_skills) == 0

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
        assert len(SKILL_TOOL_SCHEMAS) == 5

        names = {s["name"] for s in SKILL_TOOL_SCHEMAS}
        assert names == {
            "list_skills",
            "load_skill",
            "get_skill_file",
            "list_skill_links",
            "resolve_skill_link",
        }

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

    def test_list_skill_links_schema(self):
        """Test list_skill_links schema structure."""
        schema = next(s for s in SKILL_TOOL_SCHEMAS if s["name"] == "list_skill_links")

        assert "description" in schema
        assert "input_schema" in schema
        assert "name" in schema["input_schema"]["properties"]
        assert "name" in schema["input_schema"]["required"]

    def test_resolve_skill_link_schema(self):
        """Test resolve_skill_link schema structure."""
        schema = next(s for s in SKILL_TOOL_SCHEMAS if s["name"] == "resolve_skill_link")

        assert "description" in schema
        assert "input_schema" in schema
        assert "name" in schema["input_schema"]["properties"]
        assert "target" in schema["input_schema"]["properties"]
        assert set(schema["input_schema"]["required"]) == {"name", "target"}


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


class TestSkillLink:
    """Tests for SkillLink dataclass."""

    def test_skill_link_creation(self):
        """Test creating a SkillLink."""
        link = SkillLink(
            text="indicators",
            target="references/indicators.md",
            is_url=False,
            line_number=10,
        )

        assert link.text == "indicators"
        assert link.target == "references/indicators.md"
        assert link.is_url is False
        assert link.line_number == 10

    def test_skill_link_to_dict(self):
        """Test SkillLink serialization."""
        link = SkillLink(
            text="API docs",
            target="https://example.com/docs.md",
            is_url=True,
            line_number=5,
        )

        d = link.to_dict()
        assert d["text"] == "API docs"
        assert d["target"] == "https://example.com/docs.md"
        assert d["is_url"] is True
        assert d["line_number"] == 5


class TestLinkParsing:
    """Tests for link parsing functionality."""

    def test_parse_links_relative(self):
        """Test parsing relative links."""
        manager = SkillManager()

        content = """
# My Skill

See the [indicator definitions](references/indicators.md) for details.
Also check [examples](examples/basic.md).
"""
        links = manager._parse_links(content)

        assert len(links) == 2
        assert links[0].text == "indicator definitions"
        assert links[0].target == "references/indicators.md"
        assert links[0].is_url is False

        assert links[1].text == "examples"
        assert links[1].target == "examples/basic.md"
        assert links[1].is_url is False

    def test_parse_links_urls(self):
        """Test parsing URL links."""
        manager = SkillManager()

        content = """
# My Skill

Check [the docs](https://example.com/docs.md) for more info.
See [API reference](http://api.example.com/reference).
"""
        links = manager._parse_links(content)

        assert len(links) == 2
        assert links[0].text == "the docs"
        assert links[0].target == "https://example.com/docs.md"
        assert links[0].is_url is True

        assert links[1].text == "API reference"
        assert links[1].target == "http://api.example.com/reference"
        assert links[1].is_url is True

    def test_parse_links_mixed(self):
        """Test parsing mixed relative and URL links."""
        manager = SkillManager()

        content = """
# My Skill

See [local file](local/file.md) and [remote docs](https://example.com/docs).
"""
        links = manager._parse_links(content)

        assert len(links) == 2
        assert links[0].is_url is False
        assert links[1].is_url is True

    def test_parse_links_skips_anchors(self):
        """Test that anchor-only links are skipped."""
        manager = SkillManager()

        content = """
# My Skill

Jump to [section](#section-name).
See [real link](file.md).
"""
        links = manager._parse_links(content)

        assert len(links) == 1
        assert links[0].target == "file.md"

    def test_parse_links_skips_images(self):
        """Test that image links are skipped."""
        manager = SkillManager()

        content = """
# My Skill

![screenshot](images/screenshot.png)
See [documentation](docs.md).
"""
        links = manager._parse_links(content)

        assert len(links) == 1
        assert links[0].target == "docs.md"

    def test_parse_links_skips_mailto(self):
        """Test that mailto links are skipped."""
        manager = SkillManager()

        content = """
# My Skill

Contact [support](mailto:support@example.com).
See [docs](docs.md).
"""
        links = manager._parse_links(content)

        assert len(links) == 1
        assert links[0].target == "docs.md"

    def test_parse_links_tracks_line_numbers(self):
        """Test that line numbers are tracked correctly."""
        manager = SkillManager()

        content = """Line 1
Line 2
[link on line 3](file1.md)
Line 4
[link on line 5](file2.md)
"""
        links = manager._parse_links(content)

        assert len(links) == 2
        assert links[0].line_number == 3
        assert links[1].line_number == 5

    def test_skill_includes_parsed_links(self, tmp_path):
        """Test that loaded skills include parsed links."""
        skill_dir = tmp_path / "link-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: link-skill
description: A skill with links
---

# Link Skill

See [indicators](references/indicators.md) for details.
Check [API docs](https://example.com/api).
""")

        manager = SkillManager(additional_paths=[tmp_path])
        skill = manager.get_skill("link-skill")

        assert len(skill.links) == 2
        assert skill.links[0].target == "references/indicators.md"
        assert skill.links[0].is_url is False
        assert skill.links[1].target == "https://example.com/api"
        assert skill.links[1].is_url is True


class TestLinkResolution:
    """Tests for lazy link resolution."""

    def test_resolve_relative_link(self, tmp_path):
        """Test resolving a relative link."""
        skill_dir = tmp_path / "resolve-skill"
        skill_dir.mkdir()
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: resolve-skill
description: A skill to test resolution
---

See [indicators](references/indicators.md).
""")
        (refs_dir / "indicators.md").write_text("# Indicators\n\nIndicator content here.")

        manager = SkillManager(additional_paths=[tmp_path])
        content = manager.resolve_skill_link("resolve-skill", "references/indicators.md")

        assert content is not None
        assert "# Indicators" in content
        assert "Indicator content here." in content

    def test_resolve_link_caching(self, tmp_path):
        """Test that resolved links are cached."""
        skill_dir = tmp_path / "cache-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: cache-skill
description: A skill to test caching
---

See [file](file.md).
""")
        (skill_dir / "file.md").write_text("Original content.")

        manager = SkillManager(additional_paths=[tmp_path])

        # First resolution
        content1 = manager.resolve_skill_link("cache-skill", "file.md")
        assert content1 == "Original content."

        # Modify file
        (skill_dir / "file.md").write_text("Modified content.")

        # Second resolution should return cached value
        content2 = manager.resolve_skill_link("cache-skill", "file.md")
        assert content2 == "Original content."

    def test_resolve_nonexistent_link(self, tmp_path):
        """Test resolving a link to a nonexistent file."""
        skill_dir = tmp_path / "missing-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: missing-skill
description: A skill with missing link
---

See [missing](missing.md).
""")

        manager = SkillManager(additional_paths=[tmp_path])
        content = manager.resolve_skill_link("missing-skill", "missing.md")

        assert content is None

    def test_list_skill_links_via_tools(self, tmp_path):
        """Test listing skill links via tool interface."""
        skill_dir = tmp_path / "list-links-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: list-links-skill
description: A skill with multiple links
---

See [file1](file1.md) and [file2](file2.md).
Also [external](https://example.com).
""")

        manager = SkillManager(additional_paths=[tmp_path])
        tools = SkillDiscoveryTools(manager)

        result = tools.list_skill_links("list-links-skill")

        assert "error" not in result
        assert result["skill"] == "list-links-skill"
        assert len(result["links"]) == 3

    def test_resolve_skill_link_via_tools(self, tmp_path):
        """Test resolving skill link via tool interface."""
        skill_dir = tmp_path / "tool-resolve-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: tool-resolve-skill
description: A skill to test tool resolution
---

See [data](data.md).
""")
        (skill_dir / "data.md").write_text("Data content.")

        manager = SkillManager(additional_paths=[tmp_path])
        tools = SkillDiscoveryTools(manager)

        result = tools.resolve_skill_link("tool-resolve-skill", "data.md")

        assert "error" not in result
        assert result["skill"] == "tool-resolve-skill"
        assert result["target"] == "data.md"
        assert "Data content." in result["content"]

    def test_resolve_skill_link_error(self, tmp_path):
        """Test error when resolving nonexistent link via tools."""
        skill_dir = tmp_path / "error-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: error-skill
description: A skill to test error handling
---

See [missing](missing.md).
""")

        manager = SkillManager(additional_paths=[tmp_path])
        tools = SkillDiscoveryTools(manager)

        result = tools.resolve_skill_link("error-skill", "missing.md")

        assert "error" in result

    def test_load_skill_includes_links(self, tmp_path):
        """Test that load_skill includes discovered links."""
        skill_dir = tmp_path / "links-in-load"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: links-in-load
description: A skill where load includes links
---

See [ref1](ref1.md) and [external](https://example.com).
""")

        manager = SkillManager(additional_paths=[tmp_path])
        tools = SkillDiscoveryTools(manager)

        result = tools.load_skill("links-in-load")

        assert "links" in result
        assert len(result["links"]) == 2
        assert result["links"][0]["target"] == "ref1.md"
        assert result["links"][0]["is_url"] is False
        assert result["links"][1]["target"] == "https://example.com"
        assert result["links"][1]["is_url"] is True


class TestFromConfig:
    """Tests for SkillManager.from_config factory method."""

    def test_from_config_with_paths(self, tmp_path):
        """Test creating SkillManager from config with paths."""
        # Create a mock config object
        config = MagicMock()
        config.skills = MagicMock()
        config.skills.paths = [str(tmp_path / "custom-skills"), "~/other-skills"]

        manager = SkillManager.from_config(config)

        # Check that paths were expanded and added
        assert tmp_path / "custom-skills" in manager.search_paths
        assert Path.home() / "other-skills" in manager.search_paths
        # Default paths should still be there
        assert Path(".constat/skills") in manager.search_paths

    def test_from_config_no_skills_config(self):
        """Test creating SkillManager from config without skills section."""
        config = MagicMock()
        config.skills = None

        manager = SkillManager.from_config(config)

        # Should just have default paths
        assert Path(".constat/skills") in manager.search_paths
        assert Path.home() / ".constat/skills" in manager.search_paths
        assert len(manager.search_paths) == 2

    def test_from_config_empty_paths(self):
        """Test creating SkillManager from config with empty paths."""
        config = MagicMock()
        config.skills = MagicMock()
        config.skills.paths = []

        manager = SkillManager.from_config(config)

        # Should just have default paths
        assert len(manager.search_paths) == 2

    def test_from_config_tilde_expansion(self, tmp_path):
        """Test that ~ is expanded in config paths."""
        config = MagicMock()
        config.skills = MagicMock()
        config.skills.paths = ["~/my-skills"]

        manager = SkillManager.from_config(config)

        # ~ should be expanded
        assert Path.home() / "my-skills" in manager.search_paths
