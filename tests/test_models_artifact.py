# Copyright (c) 2025 Kenneth Stott
# Canary: 38d1ca9b-5a69-4d97-adce-b96f15d7e23a
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for Artifact, ArtifactType, and ARTIFACT_MIME_TYPES models."""

from __future__ import annotations
import json
import pytest
from constat.core.models import ArtifactType, ARTIFACT_MIME_TYPES, Artifact


class TestArtifactType:
    """Tests for ArtifactType enum."""

    def test_all_artifact_types_exist(self):
        """Test that all expected artifact types are defined."""
        assert ArtifactType.CODE.value == "code"
        assert ArtifactType.OUTPUT.value == "output"
        assert ArtifactType.ERROR.value == "error"

        assert ArtifactType.TABLE.value == "table"
        assert ArtifactType.JSON.value == "json"

        assert ArtifactType.HTML.value == "html"
        assert ArtifactType.MARKDOWN.value == "markdown"
        assert ArtifactType.TEXT.value == "text"

        assert ArtifactType.CHART.value == "chart"
        assert ArtifactType.PLOTLY.value == "plotly"

        assert ArtifactType.SVG.value == "svg"
        assert ArtifactType.PNG.value == "png"
        assert ArtifactType.JPEG.value == "jpeg"

        assert ArtifactType.MERMAID.value == "mermaid"
        assert ArtifactType.GRAPHVIZ.value == "graphviz"
        assert ArtifactType.DIAGRAM.value == "diagram"

        assert ArtifactType.REACT.value == "react"
        assert ArtifactType.JAVASCRIPT.value == "javascript"

    def test_artifact_type_count(self):
        """Test that the total number of artifact types is as expected."""
        assert len(ArtifactType) == 18

    def test_artifact_types_are_unique(self):
        """Test that all artifact type values are unique."""
        values = [t.value for t in ArtifactType]
        assert len(values) == len(set(values)), "Duplicate artifact type values found"

    def test_artifact_type_from_string(self):
        """Test creating ArtifactType from string value."""
        assert ArtifactType("code") == ArtifactType.CODE
        assert ArtifactType("json") == ArtifactType.JSON
        assert ArtifactType("png") == ArtifactType.PNG

    def test_invalid_artifact_type_raises(self):
        """Test that invalid artifact type raises ValueError."""
        with pytest.raises(ValueError):
            ArtifactType("invalid_type")


class TestArtifactMimeTypes:
    """Tests for ARTIFACT_MIME_TYPES mapping."""

    def test_all_artifact_types_have_mime_mapping(self):
        """Test that every ArtifactType has a MIME type mapping."""
        for artifact_type in ArtifactType:
            assert artifact_type in ARTIFACT_MIME_TYPES, (
                f"Missing MIME type mapping for {artifact_type.name}"
            )

    def test_mime_mapping_completeness(self):
        """Test ARTIFACT_MIME_TYPES has exactly as many entries as ArtifactType."""
        assert len(ARTIFACT_MIME_TYPES) == len(ArtifactType), (
            "ARTIFACT_MIME_TYPES and ArtifactType have different sizes"
        )

    @pytest.mark.parametrize("artifact_type,expected_mime", [
        ("CODE", "text/x-python"),
        ("OUTPUT", "text/plain"),
        ("ERROR", "text/plain"),
        ("TABLE", "application/json"),
        ("JSON", "application/json"),
        ("HTML", "text/html"),
        ("MARKDOWN", "text/markdown"),
        ("TEXT", "text/plain"),
        ("CHART", "application/vnd.vega.v5+json"),
        ("PLOTLY", "application/vnd.plotly.v1+json"),
        ("SVG", "image/svg+xml"),
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),
        ("MERMAID", "text/x-mermaid"),
        ("GRAPHVIZ", "text/vnd.graphviz"),
        ("DIAGRAM", "text/plain"),
        ("REACT", "text/jsx"),
        ("JAVASCRIPT", "text/javascript"),
    ])
    def test_specific_mime_type_mapping(self, artifact_type, expected_mime):
        """Test each artifact type maps to the correct MIME type."""
        art_type = getattr(ArtifactType, artifact_type)
        assert ARTIFACT_MIME_TYPES[art_type] == expected_mime

    def test_all_mime_types_are_valid_format(self):
        """Test all MIME types follow the type/subtype format."""
        for artifact_type, mime_type in ARTIFACT_MIME_TYPES.items():
            assert "/" in mime_type, (
                f"Invalid MIME type format for {artifact_type.name}: {mime_type}"
            )
            parts = mime_type.split("/")
            assert len(parts) == 2, (
                f"MIME type should have exactly one '/': {mime_type}"
            )
            assert parts[0] in {"text", "image", "application"}, (
                f"Unexpected MIME type category for {artifact_type.name}: {parts[0]}"
            )


class TestArtifact:
    """Tests for Artifact dataclass."""

    @pytest.fixture
    def sample_artifact(self):
        """Create a sample artifact for testing."""
        return Artifact(
            id=1,
            name="test_chart",
            artifact_type=ArtifactType.CHART,
            content='{"$schema": "https://vega.github.io/schema/vega-lite/v5.json"}',
            step_number=2,
            attempt=1,
            title="Sales Chart",
            description="A chart showing sales trends",
        )

    def test_artifact_creation(self, sample_artifact):
        """Test creating an artifact with all fields."""
        assert sample_artifact.id == 1
        assert sample_artifact.name == "test_chart"
        assert sample_artifact.artifact_type == ArtifactType.CHART
        assert "$schema" in sample_artifact.content
        assert sample_artifact.step_number == 2
        assert sample_artifact.attempt == 1
        assert sample_artifact.title == "Sales Chart"
        assert sample_artifact.description == "A chart showing sales trends"

    def test_artifact_defaults(self):
        """Test artifact default values."""
        artifact = Artifact(
            id=1,
            name="simple",
            artifact_type=ArtifactType.TEXT,
            content="Hello",
        )

        assert artifact.step_number == 0
        assert artifact.attempt == 1
        assert artifact.title is None
        assert artifact.description is None
        assert artifact.content_type is None
        assert artifact.metadata == {}
        assert artifact.created_at is None

    def test_mime_type_property_returns_mapped_type(self):
        """Test mime_type property returns correct MIME type from mapping."""
        artifact = Artifact(
            id=1,
            name="test",
            artifact_type=ArtifactType.HTML,
            content="<html></html>",
        )

        assert artifact.mime_type == "text/html"

    def test_mime_type_property_with_content_type_override(self):
        """Test mime_type property respects content_type override."""
        artifact = Artifact(
            id=1,
            name="test",
            artifact_type=ArtifactType.HTML,
            content="<html></html>",
            content_type="application/xhtml+xml",
        )

        assert artifact.mime_type == "application/xhtml+xml"

    @pytest.mark.parametrize("artifact_type,expected_mime", [
        ("CODE", "text/x-python"),
        ("JSON", "application/json"),
        ("PNG", "image/png"),
        ("SVG", "image/svg+xml"),
        ("PLOTLY", "application/vnd.plotly.v1+json"),
    ])
    def test_mime_type_for_various_types(self, artifact_type, expected_mime):
        """Test mime_type property for various artifact types."""
        art_type = getattr(ArtifactType, artifact_type)
        artifact = Artifact(
            id=1,
            name="test",
            artifact_type=art_type,
            content="test content",
        )

        assert artifact.mime_type == expected_mime

    def test_is_binary_for_png(self):
        """Test is_binary returns True for PNG artifacts."""
        artifact = Artifact(
            id=1,
            name="image",
            artifact_type=ArtifactType.PNG,
            content="base64encodedcontent",
        )

        assert artifact.is_binary is True

    def test_is_binary_for_jpeg(self):
        """Test is_binary returns True for JPEG artifacts."""
        artifact = Artifact(
            id=1,
            name="image",
            artifact_type=ArtifactType.JPEG,
            content="base64encodedcontent",
        )

        assert artifact.is_binary is True

    @pytest.mark.parametrize("artifact_type", [
        "CODE", "OUTPUT", "ERROR", "TABLE", "JSON", "HTML",
        "MARKDOWN", "TEXT", "CHART", "PLOTLY", "SVG",
        "MERMAID", "GRAPHVIZ", "DIAGRAM", "REACT", "JAVASCRIPT",
    ])
    def test_is_binary_false_for_text_types(self, artifact_type):
        """Test is_binary returns False for non-binary artifact types."""
        art_type = getattr(ArtifactType, artifact_type)
        artifact = Artifact(
            id=1,
            name="test",
            artifact_type=art_type,
            content="test content",
        )

        assert artifact.is_binary is False

    def test_to_dict_contains_all_fields(self, sample_artifact):
        """Test to_dict returns all expected fields."""
        result = sample_artifact.to_dict()

        assert "id" in result
        assert "name" in result
        assert "type" in result
        assert "content" in result
        assert "step_number" in result
        assert "attempt" in result
        assert "title" in result
        assert "description" in result
        assert "content_type" in result
        assert "metadata" in result
        assert "created_at" in result

    def test_to_dict_values(self, sample_artifact):
        """Test to_dict returns correct values."""
        result = sample_artifact.to_dict()

        assert result["id"] == 1
        assert result["name"] == "test_chart"
        assert result["type"] == "chart"
        assert result["step_number"] == 2
        assert result["attempt"] == 1
        assert result["title"] == "Sales Chart"
        assert result["description"] == "A chart showing sales trends"
        assert result["content_type"] == "application/vnd.vega.v5+json"

    def test_to_dict_with_content_type_override(self):
        """Test to_dict uses overridden content_type."""
        artifact = Artifact(
            id=1,
            name="test",
            artifact_type=ArtifactType.JSON,
            content="{}",
            content_type="application/vnd.custom+json",
        )

        result = artifact.to_dict()
        assert result["content_type"] == "application/vnd.custom+json"

    def test_to_dict_with_metadata(self):
        """Test to_dict includes metadata."""
        artifact = Artifact(
            id=1,
            name="test",
            artifact_type=ArtifactType.CHART,
            content="{}",
            metadata={"width": 800, "height": 600, "theme": "dark"},
        )

        result = artifact.to_dict()
        assert result["metadata"] == {"width": 800, "height": 600, "theme": "dark"}

    def test_to_dict_with_created_at(self):
        """Test to_dict includes created_at timestamp."""
        artifact = Artifact(
            id=1,
            name="test",
            artifact_type=ArtifactType.TEXT,
            content="Hello",
            created_at="2024-01-15T10:30:00Z",
        )

        result = artifact.to_dict()
        assert result["created_at"] == "2024-01-15T10:30:00Z"

    def test_to_dict_is_json_serializable(self, sample_artifact):
        """Test to_dict output can be serialized to JSON."""
        result = sample_artifact.to_dict()

        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        restored = json.loads(json_str)
        assert restored == result

    def test_artifact_with_empty_content(self):
        """Test artifact can be created with empty content."""
        artifact = Artifact(
            id=1,
            name="empty",
            artifact_type=ArtifactType.TEXT,
            content="",
        )

        assert artifact.content == ""
        assert artifact.mime_type == "text/plain"

    def test_artifact_with_unicode_content(self):
        """Test artifact handles unicode content correctly."""
        content = "Hello, \u4e16\u754c! \U0001F600"
        artifact = Artifact(
            id=1,
            name="unicode_test",
            artifact_type=ArtifactType.TEXT,
            content=content,
        )

        assert artifact.content == content
        result = artifact.to_dict()
        assert result["content"] == content

    def test_artifact_with_special_characters_in_name(self):
        """Test artifact handles special characters in name."""
        artifact = Artifact(
            id=1,
            name="test-chart_2024.01.15",
            artifact_type=ArtifactType.CHART,
            content="{}",
        )

        assert artifact.name == "test-chart_2024.01.15"
        result = artifact.to_dict()
        assert result["name"] == "test-chart_2024.01.15"


class TestArtifactEdgeCases:
    """Edge case and boundary tests for Artifact."""

    def test_artifact_with_large_content(self):
        """Test artifact handles large content."""
        large_content = "x" * (1024 * 1024)
        artifact = Artifact(
            id=1,
            name="large",
            artifact_type=ArtifactType.TEXT,
            content=large_content,
        )

        assert len(artifact.content) == 1024 * 1024
        result = artifact.to_dict()
        assert len(result["content"]) == 1024 * 1024

    def test_artifact_with_zero_id(self):
        """Test artifact with id of 0."""
        artifact = Artifact(
            id=0,
            name="zero",
            artifact_type=ArtifactType.TEXT,
            content="test",
        )

        assert artifact.id == 0
        result = artifact.to_dict()
        assert result["id"] == 0

    def test_artifact_with_negative_id(self):
        """Test artifact with negative id (unusual but valid dataclass)."""
        artifact = Artifact(
            id=-1,
            name="negative",
            artifact_type=ArtifactType.TEXT,
            content="test",
        )

        assert artifact.id == -1

    def test_artifact_with_step_number_zero(self):
        """Test artifact with step_number of 0 (default)."""
        artifact = Artifact(
            id=1,
            name="test",
            artifact_type=ArtifactType.TEXT,
            content="test",
            step_number=0,
        )

        assert artifact.step_number == 0

    def test_artifact_with_attempt_zero(self):
        """Test artifact with attempt of 0."""
        artifact = Artifact(
            id=1,
            name="test",
            artifact_type=ArtifactType.TEXT,
            content="test",
            attempt=0,
        )

        assert artifact.attempt == 0

    def test_artifact_with_multiline_content(self):
        """Test artifact handles multiline content."""
        content = """Line 1
Line 2
Line 3
"""
        artifact = Artifact(
            id=1,
            name="multiline",
            artifact_type=ArtifactType.CODE,
            content=content,
        )

        assert artifact.content == content
        assert artifact.content.count("\n") == 3

    def test_artifact_with_empty_metadata(self):
        """Test artifact with explicitly empty metadata."""
        artifact = Artifact(
            id=1,
            name="test",
            artifact_type=ArtifactType.TEXT,
            content="test",
            metadata={},
        )

        assert artifact.metadata == {}
        result = artifact.to_dict()
        assert result["metadata"] == {}

    def test_artifact_with_nested_metadata(self):
        """Test artifact with nested metadata structure."""
        nested_metadata = {
            "dimensions": {"width": 800, "height": 600},
            "options": {"theme": "dark", "interactive": True},
            "tags": ["chart", "visualization"],
        }

        artifact = Artifact(
            id=1,
            name="test",
            artifact_type=ArtifactType.CHART,
            content="{}",
            metadata=nested_metadata,
        )

        assert artifact.metadata == nested_metadata
        result = artifact.to_dict()
        assert result["metadata"]["dimensions"]["width"] == 800
