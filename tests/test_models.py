"""Tests for core data models."""

import pytest
from constat.core.models import Plan, Step, StepResult, StepStatus, StepType


class TestStep:
    """Tests for Step model."""

    def test_step_creation(self):
        """Test creating a step."""
        step = Step(
            number=1,
            goal="Load customer data",
            expected_inputs=["config"],
            expected_outputs=["customers_df"],
        )

        assert step.number == 1
        assert step.goal == "Load customer data"
        assert step.expected_inputs == ["config"]
        assert step.expected_outputs == ["customers_df"]
        assert step.step_type == StepType.PYTHON
        assert step.status == StepStatus.PENDING
        assert step.code is None
        assert step.result is None

    def test_step_defaults(self):
        """Test step defaults."""
        step = Step(number=1, goal="Test step")

        assert step.expected_inputs == []
        assert step.expected_outputs == []
        assert step.step_type == StepType.PYTHON


class TestStepResult:
    """Tests for StepResult model."""

    def test_successful_result(self):
        """Test successful step result."""
        result = StepResult(
            success=True,
            stdout="Loaded 100 rows",
            attempts=1,
            duration_ms=500,
        )

        assert result.success
        assert result.stdout == "Loaded 100 rows"
        assert result.error is None
        assert result.attempts == 1
        assert result.duration_ms == 500

    def test_failed_result(self):
        """Test failed step result."""
        result = StepResult(
            success=False,
            stdout="",
            error="Column not found",
            attempts=3,
            duration_ms=1500,
        )

        assert not result.success
        assert result.error == "Column not found"
        assert result.attempts == 3


class TestPlan:
    """Tests for Plan model."""

    @pytest.fixture
    def sample_plan(self):
        """Create a sample plan for testing."""
        return Plan(
            problem="Analyze sales data",
            steps=[
                Step(number=1, goal="Load data", expected_outputs=["sales_df"]),
                Step(number=2, goal="Calculate totals", expected_inputs=["sales_df"], expected_outputs=["totals"]),
                Step(number=3, goal="Generate report", expected_inputs=["totals"]),
            ],
        )

    def test_plan_creation(self, sample_plan):
        """Test plan creation."""
        assert sample_plan.problem == "Analyze sales data"
        assert len(sample_plan.steps) == 3
        assert sample_plan.current_step == 0
        assert sample_plan.completed_steps == []

    def test_is_complete_initially_false(self, sample_plan):
        """Test is_complete is False initially."""
        assert not sample_plan.is_complete

    def test_next_step(self, sample_plan):
        """Test getting next step."""
        next_step = sample_plan.next_step
        assert next_step is not None
        assert next_step.number == 1

    def test_get_step(self, sample_plan):
        """Test getting step by number."""
        step = sample_plan.get_step(2)
        assert step is not None
        assert step.goal == "Calculate totals"

        assert sample_plan.get_step(99) is None

    def test_mark_step_completed(self, sample_plan):
        """Test marking a step as completed."""
        result = StepResult(success=True, stdout="Done", attempts=1, duration_ms=100)
        sample_plan.mark_step_completed(1, result)

        assert 1 in sample_plan.completed_steps
        step = sample_plan.get_step(1)
        assert step.status == StepStatus.COMPLETED
        assert step.result == result

    def test_mark_step_failed(self, sample_plan):
        """Test marking a step as failed."""
        result = StepResult(success=False, stdout="", error="Error", attempts=3, duration_ms=500)
        sample_plan.mark_step_failed(2, result)

        assert 2 in sample_plan.failed_steps
        step = sample_plan.get_step(2)
        assert step.status == StepStatus.FAILED

    def test_is_complete_when_all_done(self, sample_plan):
        """Test is_complete when all steps are done."""
        result = StepResult(success=True, stdout="Done", attempts=1, duration_ms=100)

        sample_plan.mark_step_completed(1, result)
        assert not sample_plan.is_complete

        sample_plan.mark_step_completed(2, result)
        assert not sample_plan.is_complete

        sample_plan.mark_step_completed(3, result)
        assert sample_plan.is_complete

    def test_next_step_skips_completed(self, sample_plan):
        """Test next_step skips completed steps."""
        result = StepResult(success=True, stdout="Done", attempts=1, duration_ms=100)
        sample_plan.mark_step_completed(1, result)

        next_step = sample_plan.next_step
        assert next_step.number == 2


class TestArtifactType:
    """Tests for ArtifactType enum."""

    def test_all_artifact_types_exist(self):
        """Test that all expected artifact types are defined."""
        from constat.core.models import ArtifactType

        # Code and execution artifacts
        assert ArtifactType.CODE.value == "code"
        assert ArtifactType.OUTPUT.value == "output"
        assert ArtifactType.ERROR.value == "error"

        # Data artifacts
        assert ArtifactType.TABLE.value == "table"
        assert ArtifactType.JSON.value == "json"

        # Rich content artifacts
        assert ArtifactType.HTML.value == "html"
        assert ArtifactType.MARKDOWN.value == "markdown"
        assert ArtifactType.TEXT.value == "text"

        # Chart/visualization artifacts
        assert ArtifactType.CHART.value == "chart"
        assert ArtifactType.PLOTLY.value == "plotly"

        # Image artifacts
        assert ArtifactType.SVG.value == "svg"
        assert ArtifactType.PNG.value == "png"
        assert ArtifactType.JPEG.value == "jpeg"

        # Diagram artifacts
        assert ArtifactType.MERMAID.value == "mermaid"
        assert ArtifactType.GRAPHVIZ.value == "graphviz"
        assert ArtifactType.DIAGRAM.value == "diagram"

        # Interactive artifacts
        assert ArtifactType.REACT.value == "react"
        assert ArtifactType.JAVASCRIPT.value == "javascript"

    def test_artifact_type_count(self):
        """Test that the total number of artifact types is as expected."""
        from constat.core.models import ArtifactType

        # This helps catch if someone adds a new type but forgets tests
        assert len(ArtifactType) == 18

    def test_artifact_types_are_unique(self):
        """Test that all artifact type values are unique."""
        from constat.core.models import ArtifactType

        values = [t.value for t in ArtifactType]
        assert len(values) == len(set(values)), "Duplicate artifact type values found"

    def test_artifact_type_from_string(self):
        """Test creating ArtifactType from string value."""
        from constat.core.models import ArtifactType

        assert ArtifactType("code") == ArtifactType.CODE
        assert ArtifactType("json") == ArtifactType.JSON
        assert ArtifactType("png") == ArtifactType.PNG

    def test_invalid_artifact_type_raises(self):
        """Test that invalid artifact type raises ValueError."""
        from constat.core.models import ArtifactType

        with pytest.raises(ValueError):
            ArtifactType("invalid_type")


class TestArtifactMimeTypes:
    """Tests for ARTIFACT_MIME_TYPES mapping."""

    def test_all_artifact_types_have_mime_mapping(self):
        """Test that every ArtifactType has a MIME type mapping."""
        from constat.core.models import ArtifactType, ARTIFACT_MIME_TYPES

        for artifact_type in ArtifactType:
            assert artifact_type in ARTIFACT_MIME_TYPES, (
                f"Missing MIME type mapping for {artifact_type.name}"
            )

    def test_mime_mapping_completeness(self):
        """Test ARTIFACT_MIME_TYPES has exactly as many entries as ArtifactType."""
        from constat.core.models import ArtifactType, ARTIFACT_MIME_TYPES

        assert len(ARTIFACT_MIME_TYPES) == len(ArtifactType), (
            "ARTIFACT_MIME_TYPES and ArtifactType have different sizes"
        )

    @pytest.mark.parametrize("artifact_type,expected_mime", [
        # Code and execution
        ("CODE", "text/x-python"),
        ("OUTPUT", "text/plain"),
        ("ERROR", "text/plain"),
        # Data
        ("TABLE", "application/json"),
        ("JSON", "application/json"),
        # Rich content
        ("HTML", "text/html"),
        ("MARKDOWN", "text/markdown"),
        ("TEXT", "text/plain"),
        # Charts
        ("CHART", "application/vnd.vega.v5+json"),
        ("PLOTLY", "application/vnd.plotly.v1+json"),
        # Images
        ("SVG", "image/svg+xml"),
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),
        # Diagrams
        ("MERMAID", "text/x-mermaid"),
        ("GRAPHVIZ", "text/vnd.graphviz"),
        ("DIAGRAM", "text/plain"),
        # Interactive
        ("REACT", "text/jsx"),
        ("JAVASCRIPT", "text/javascript"),
    ])
    def test_specific_mime_type_mapping(self, artifact_type, expected_mime):
        """Test each artifact type maps to the correct MIME type."""
        from constat.core.models import ArtifactType, ARTIFACT_MIME_TYPES

        art_type = getattr(ArtifactType, artifact_type)
        assert ARTIFACT_MIME_TYPES[art_type] == expected_mime

    def test_all_mime_types_are_valid_format(self):
        """Test all MIME types follow the type/subtype format."""
        from constat.core.models import ARTIFACT_MIME_TYPES

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
        from constat.core.models import Artifact, ArtifactType

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
        from constat.core.models import ArtifactType

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
        from constat.core.models import Artifact, ArtifactType

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
        from constat.core.models import Artifact, ArtifactType

        artifact = Artifact(
            id=1,
            name="test",
            artifact_type=ArtifactType.HTML,
            content="<html></html>",
        )

        assert artifact.mime_type == "text/html"

    def test_mime_type_property_with_content_type_override(self):
        """Test mime_type property respects content_type override."""
        from constat.core.models import Artifact, ArtifactType

        artifact = Artifact(
            id=1,
            name="test",
            artifact_type=ArtifactType.HTML,
            content="<html></html>",
            content_type="application/xhtml+xml",  # Override
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
        from constat.core.models import Artifact, ArtifactType

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
        from constat.core.models import Artifact, ArtifactType

        artifact = Artifact(
            id=1,
            name="image",
            artifact_type=ArtifactType.PNG,
            content="base64encodedcontent",
        )

        assert artifact.is_binary is True

    def test_is_binary_for_jpeg(self):
        """Test is_binary returns True for JPEG artifacts."""
        from constat.core.models import Artifact, ArtifactType

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
        from constat.core.models import Artifact, ArtifactType

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
        assert result["type"] == "chart"  # Note: uses .value
        assert result["step_number"] == 2
        assert result["attempt"] == 1
        assert result["title"] == "Sales Chart"
        assert result["description"] == "A chart showing sales trends"
        # content_type uses mime_type property
        assert result["content_type"] == "application/vnd.vega.v5+json"

    def test_to_dict_with_content_type_override(self):
        """Test to_dict uses overridden content_type."""
        from constat.core.models import Artifact, ArtifactType

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
        from constat.core.models import Artifact, ArtifactType

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
        from constat.core.models import Artifact, ArtifactType

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
        import json

        result = sample_artifact.to_dict()

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should roundtrip
        restored = json.loads(json_str)
        assert restored == result

    def test_artifact_with_empty_content(self):
        """Test artifact can be created with empty content."""
        from constat.core.models import Artifact, ArtifactType

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
        from constat.core.models import Artifact, ArtifactType

        content = "Hello, \u4e16\u754c! \U0001F600"  # "Hello, World!" with emoji
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
        from constat.core.models import Artifact, ArtifactType

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
        from constat.core.models import Artifact, ArtifactType

        # 1MB of content
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
        from constat.core.models import Artifact, ArtifactType

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
        from constat.core.models import Artifact, ArtifactType

        artifact = Artifact(
            id=-1,
            name="negative",
            artifact_type=ArtifactType.TEXT,
            content="test",
        )

        assert artifact.id == -1

    def test_artifact_with_step_number_zero(self):
        """Test artifact with step_number of 0 (default)."""
        from constat.core.models import Artifact, ArtifactType

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
        from constat.core.models import Artifact, ArtifactType

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
        from constat.core.models import Artifact, ArtifactType

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
        from constat.core.models import Artifact, ArtifactType

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
        from constat.core.models import Artifact, ArtifactType

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
