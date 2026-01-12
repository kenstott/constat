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


class TestPlanDependencies:
    """Tests for Plan dependency analysis and parallel execution support."""

    def test_step_depends_on_field(self):
        """Test that Step has depends_on field."""
        step = Step(
            number=1,
            goal="Test step",
            depends_on=[2, 3],
        )
        assert step.depends_on == [2, 3]

    def test_step_depends_on_defaults_empty(self):
        """Test depends_on defaults to empty list."""
        step = Step(number=1, goal="Test step")
        assert step.depends_on == []

    def test_infer_dependencies_from_inputs_outputs(self):
        """Test inferring dependencies from input/output overlap."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="Load A", expected_outputs=["data_a"]),
                Step(number=2, goal="Load B", expected_outputs=["data_b"]),
                Step(number=3, goal="Combine", expected_inputs=["data_a", "data_b"], expected_outputs=["combined"]),
                Step(number=4, goal="Report", expected_inputs=["combined"]),
            ],
        )

        plan.infer_dependencies()

        # Step 3 should depend on steps 1 and 2
        step3 = plan.get_step(3)
        assert 1 in step3.depends_on
        assert 2 in step3.depends_on

        # Step 4 should depend on step 3
        step4 = plan.get_step(4)
        assert 3 in step4.depends_on

        # Steps 1 and 2 have no dependencies
        assert plan.get_step(1).depends_on == []
        assert plan.get_step(2).depends_on == []

    def test_get_dependency_graph(self):
        """Test getting dependency graph as adjacency list."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[]),
                Step(number=3, goal="C", depends_on=[1, 2]),
            ],
        )

        graph = plan.get_dependency_graph()

        assert graph == {1: [], 2: [], 3: [1, 2]}

    def test_get_runnable_steps_initial(self):
        """Test getting runnable steps when none are completed."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[]),
                Step(number=3, goal="C", depends_on=[1, 2]),
            ],
        )

        runnable = plan.get_runnable_steps()

        # Steps 1 and 2 should be runnable (no dependencies)
        runnable_numbers = [s.number for s in runnable]
        assert 1 in runnable_numbers
        assert 2 in runnable_numbers
        assert 3 not in runnable_numbers

    def test_get_runnable_steps_after_completion(self):
        """Test getting runnable steps after some are completed."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[]),
                Step(number=3, goal="C", depends_on=[1, 2]),
            ],
        )

        result = StepResult(success=True, stdout="Done", attempts=1, duration_ms=100)
        plan.mark_step_completed(1, result)
        plan.mark_step_completed(2, result)

        runnable = plan.get_runnable_steps()

        # Step 3 should now be runnable
        runnable_numbers = [s.number for s in runnable]
        assert 3 in runnable_numbers
        assert 1 not in runnable_numbers  # Already completed
        assert 2 not in runnable_numbers  # Already completed

    def test_get_runnable_steps_partial_completion(self):
        """Test getting runnable steps with partial completion."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[]),
                Step(number=3, goal="C", depends_on=[1, 2]),
            ],
        )

        result = StepResult(success=True, stdout="Done", attempts=1, duration_ms=100)
        plan.mark_step_completed(1, result)  # Only 1 completed, not 2

        runnable = plan.get_runnable_steps()

        # Step 3 should NOT be runnable (step 2 not complete)
        runnable_numbers = [s.number for s in runnable]
        assert 2 in runnable_numbers  # Still pending
        assert 3 not in runnable_numbers  # Dependencies not satisfied

    def test_get_execution_order_waves(self):
        """Test getting execution order as waves."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[]),
                Step(number=3, goal="C", depends_on=[1]),
                Step(number=4, goal="D", depends_on=[2]),
                Step(number=5, goal="E", depends_on=[3, 4]),
            ],
        )

        waves = plan.get_execution_order()

        # Wave 1: steps 1, 2 (no dependencies)
        # Wave 2: steps 3, 4 (depend on wave 1)
        # Wave 3: step 5 (depends on wave 2)
        assert len(waves) == 3
        assert set(waves[0]) == {1, 2}
        assert set(waves[1]) == {3, 4}
        assert set(waves[2]) == {5}

    def test_get_execution_order_sequential(self):
        """Test execution order for fully sequential plan."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[1]),
                Step(number=3, goal="C", depends_on=[2]),
            ],
        )

        waves = plan.get_execution_order()

        # Each step in its own wave
        assert waves == [[1], [2], [3]]

    def test_get_execution_order_fully_parallel(self):
        """Test execution order for fully parallel plan."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[]),
                Step(number=3, goal="C", depends_on=[]),
                Step(number=4, goal="D", depends_on=[]),
            ],
        )

        waves = plan.get_execution_order()

        # All steps in one wave
        assert len(waves) == 1
        assert set(waves[0]) == {1, 2, 3, 4}

    def test_employee_count_example(self):
        """Test the employee count across companies example from requirements."""
        # 5 companies, get employee count for each, then sum
        plan = Plan(
            problem="Get total employees across all companies",
            steps=[
                Step(number=1, goal="Get employee count from Company A", depends_on=[], expected_outputs=["count_a"]),
                Step(number=2, goal="Get employee count from Company B", depends_on=[], expected_outputs=["count_b"]),
                Step(number=3, goal="Get employee count from Company C", depends_on=[], expected_outputs=["count_c"]),
                Step(number=4, goal="Get employee count from Company D", depends_on=[], expected_outputs=["count_d"]),
                Step(number=5, goal="Get employee count from Company E", depends_on=[], expected_outputs=["count_e"]),
                Step(number=6, goal="Compute total sum", depends_on=[1, 2, 3, 4, 5], expected_inputs=["count_a", "count_b", "count_c", "count_d", "count_e"]),
            ],
        )

        waves = plan.get_execution_order()

        # Wave 1: All 5 company queries in parallel
        # Wave 2: Sum step
        assert len(waves) == 2
        assert set(waves[0]) == {1, 2, 3, 4, 5}
        assert waves[1] == [6]


# =============================================================================
# PARALLEL STEP SCHEDULER COMPONENT TESTS
# =============================================================================


class TestParallelStepSchedulerComponent:
    """Component tests for ParallelStepScheduler execution."""

    @pytest.mark.asyncio
    async def test_parallel_execution_timing_proves_concurrency(self):
        """Component: Verify parallel execution by measuring actual timing."""
        import time
        import asyncio
        from constat.execution.parallel_scheduler import ParallelStepScheduler, SchedulerConfig

        DELAY_MS = 100
        NUM_PARALLEL_STEPS = 3

        # Track execution times
        execution_log = []

        def slow_step_executor(step: Step, namespace: dict) -> StepResult:
            """Step executor that takes DELAY_MS to complete."""
            execution_log.append(("start", step.number, time.time()))
            time.sleep(DELAY_MS / 1000)
            execution_log.append(("end", step.number, time.time()))
            return StepResult(
                success=True,
                stdout=f"Step {step.number} done",
                variables={f"result_{step.number}": step.number * 10},
                duration_ms=DELAY_MS,
            )

        # Create plan with 3 parallel steps + 1 dependent step
        plan = Plan(
            problem="Test parallel execution",
            steps=[
                Step(number=1, goal="Task A", depends_on=[]),
                Step(number=2, goal="Task B", depends_on=[]),
                Step(number=3, goal="Task C", depends_on=[]),
                Step(number=4, goal="Aggregate", depends_on=[1, 2, 3]),
            ],
        )

        scheduler = ParallelStepScheduler(
            step_executor=slow_step_executor,
            config=SchedulerConfig(max_concurrent_steps=10),
        )

        start = time.time()
        result = await scheduler.execute_plan(plan)
        total_time = (time.time() - start) * 1000

        # Verify success
        assert result.success
        assert len(result.completed_steps) == 4

        # Verify wave structure
        assert len(result.execution_waves) == 2
        assert set(result.execution_waves[0]) == {1, 2, 3}
        assert result.execution_waves[1] == [4]

        # CRITICAL: If parallel, steps 1-3 should complete in ~100ms (not 300ms)
        # Total should be ~200ms (wave 1: 100ms, wave 2: 100ms)
        expected_sequential = NUM_PARALLEL_STEPS * DELAY_MS + DELAY_MS  # 400ms
        expected_parallel = 2 * DELAY_MS + 50  # ~250ms with overhead

        print(f"\nTotal time: {total_time:.0f}ms")
        print(f"Expected sequential: {expected_sequential}ms")
        print(f"Expected parallel: ~{expected_parallel}ms")

        assert total_time < expected_sequential * 0.7, (
            f"Parallel execution took {total_time:.0f}ms, expected <{expected_sequential * 0.7:.0f}ms. "
            f"Steps may not be running in parallel!"
        )

    @pytest.mark.asyncio
    async def test_dependency_ordering_enforced(self):
        """Component: Verify dependent steps wait for prerequisites."""
        import time
        from constat.execution.parallel_scheduler import ParallelStepScheduler

        completion_order = []

        def tracking_executor(step: Step, namespace: dict) -> StepResult:
            """Track when each step completes."""
            time.sleep(0.05)  # Small delay
            completion_order.append(step.number)
            return StepResult(
                success=True,
                stdout=f"Step {step.number}",
                variables={f"out_{step.number}": True},
            )

        # Chain: 1 -> 2 -> 3 (fully sequential)
        plan = Plan(
            problem="Test dependency ordering",
            steps=[
                Step(number=1, goal="First", depends_on=[]),
                Step(number=2, goal="Second", depends_on=[1]),
                Step(number=3, goal="Third", depends_on=[2]),
            ],
        )

        scheduler = ParallelStepScheduler(step_executor=tracking_executor)
        result = await scheduler.execute_plan(plan)

        assert result.success
        # Steps must complete in order due to dependencies
        assert completion_order == [1, 2, 3]
        # Each step in its own wave
        assert result.execution_waves == [[1], [2], [3]]

    @pytest.mark.asyncio
    async def test_namespace_flows_between_steps(self):
        """Component: Verify step outputs are available to dependent steps."""
        from constat.execution.parallel_scheduler import ParallelStepScheduler

        received_values = {}

        def namespace_checking_executor(step: Step, namespace: dict) -> StepResult:
            """Check namespace and produce output."""
            received_values[step.number] = dict(namespace)

            if step.number == 1:
                return StepResult(
                    success=True,
                    stdout="Step 1",
                    variables={"value_a": 100},
                )
            elif step.number == 2:
                return StepResult(
                    success=True,
                    stdout="Step 2",
                    variables={"value_b": 200},
                )
            elif step.number == 3:
                # Should see both value_a and value_b
                total = namespace.get("value_a", 0) + namespace.get("value_b", 0)
                return StepResult(
                    success=True,
                    stdout=f"Total: {total}",
                    variables={"total": total},
                )

            return StepResult(success=True, stdout="")

        plan = Plan(
            problem="Test namespace flow",
            steps=[
                Step(number=1, goal="Produce A", depends_on=[]),
                Step(number=2, goal="Produce B", depends_on=[]),
                Step(number=3, goal="Sum A+B", depends_on=[1, 2]),
            ],
        )

        scheduler = ParallelStepScheduler(step_executor=namespace_checking_executor)
        result = await scheduler.execute_plan(plan)

        assert result.success

        # Step 3 should have received both values
        assert "value_a" in received_values[3]
        assert "value_b" in received_values[3]
        assert received_values[3]["value_a"] == 100
        assert received_values[3]["value_b"] == 200

        # Final result should have computed total
        assert result.step_results[3].variables.get("total") == 300

    @pytest.mark.asyncio
    async def test_fail_fast_stops_execution(self):
        """Component: Verify fail_fast stops on first failure."""
        from constat.execution.parallel_scheduler import ParallelStepScheduler, SchedulerConfig

        executed_steps = []

        def failing_executor(step: Step, namespace: dict) -> StepResult:
            executed_steps.append(step.number)
            if step.number == 2:
                return StepResult(success=False, stdout="", error="Step 2 failed")
            return StepResult(success=True, stdout=f"Step {step.number}")

        plan = Plan(
            problem="Test fail fast",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B (fails)", depends_on=[]),
                Step(number=3, goal="C", depends_on=[1, 2]),
            ],
        )

        scheduler = ParallelStepScheduler(
            step_executor=failing_executor,
            config=SchedulerConfig(fail_fast=True),
        )
        result = await scheduler.execute_plan(plan)

        assert not result.success
        assert 2 in result.failed_steps
        # Step 3 should not execute (fail_fast + dependency on failed step)
        assert 3 not in executed_steps

    @pytest.mark.asyncio
    async def test_employee_count_scenario(self):
        """Component: Test the employee count across companies scenario."""
        import time
        from constat.execution.parallel_scheduler import ParallelStepScheduler

        DELAY_MS = 50
        step_start_times = {}

        def company_count_executor(step: Step, namespace: dict) -> StepResult:
            step_start_times[step.number] = time.time()
            time.sleep(DELAY_MS / 1000)

            if step.number <= 5:
                # Company queries - return employee count
                return StepResult(
                    success=True,
                    stdout=f"Company {step.number}: 100 employees",
                    variables={f"count_{step.number}": 100},
                )
            else:
                # Sum step
                total = sum(
                    namespace.get(f"count_{i}", 0)
                    for i in range(1, 6)
                )
                return StepResult(
                    success=True,
                    stdout=f"Total: {total} employees",
                    variables={"total": total},
                )

        plan = Plan(
            problem="Get total employees across all companies",
            steps=[
                Step(number=1, goal="Get count from Company A", depends_on=[]),
                Step(number=2, goal="Get count from Company B", depends_on=[]),
                Step(number=3, goal="Get count from Company C", depends_on=[]),
                Step(number=4, goal="Get count from Company D", depends_on=[]),
                Step(number=5, goal="Get count from Company E", depends_on=[]),
                Step(number=6, goal="Compute total", depends_on=[1, 2, 3, 4, 5]),
            ],
        )

        scheduler = ParallelStepScheduler(step_executor=company_count_executor)

        start = time.time()
        result = await scheduler.execute_plan(plan)
        total_time = (time.time() - start) * 1000

        assert result.success
        assert len(result.completed_steps) == 6

        # Verify waves: 5 parallel, then 1
        assert len(result.execution_waves) == 2
        assert set(result.execution_waves[0]) == {1, 2, 3, 4, 5}
        assert result.execution_waves[1] == [6]

        # Verify total was computed correctly
        assert result.step_results[6].variables.get("total") == 500

        # Verify parallel execution timing
        # Sequential: 6 * 50ms = 300ms
        # Parallel: 2 * 50ms = 100ms (+ overhead)
        print(f"\nEmployee count scenario: {total_time:.0f}ms")
        assert total_time < 200, f"Expected <200ms, got {total_time:.0f}ms"

        # Verify steps 1-5 started nearly simultaneously
        start_times = [step_start_times[i] for i in range(1, 6)]
        start_spread = max(start_times) - min(start_times)
        print(f"Start time spread for parallel steps: {start_spread*1000:.0f}ms")
        assert start_spread < 0.05, "Parallel steps should start within 50ms of each other"

    def test_sync_wrapper_works(self):
        """Test synchronous wrapper for execute_plan."""
        from constat.execution.parallel_scheduler import ParallelStepScheduler

        def simple_executor(step: Step, namespace: dict) -> StepResult:
            return StepResult(
                success=True,
                stdout=f"Step {step.number}",
                variables={f"v{step.number}": step.number},
            )

        plan = Plan(
            problem="Test sync",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[1]),
            ],
        )

        scheduler = ParallelStepScheduler(step_executor=simple_executor)
        result = scheduler.execute_plan_sync(plan)

        assert result.success
        assert len(result.completed_steps) == 2
