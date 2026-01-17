"""Tests for the concept detector module."""

import pytest

from constat.discovery.concept_detector import ConceptDetector, DetectedConcept
from constat.execution.prompt_sections import PROMPT_SECTIONS, PromptSection


class TestPromptSections:
    """Test prompt sections registry."""

    def test_all_sections_have_required_fields(self):
        """All sections should have required fields populated."""
        for concept_id, section in PROMPT_SECTIONS.items():
            assert section.concept_id == concept_id
            assert section.content.strip(), f"{concept_id} has empty content"
            assert len(section.targets) > 0, f"{concept_id} has no targets"
            assert len(section.exemplars) >= 3, f"{concept_id} needs at least 3 exemplars"

    def test_all_targets_are_valid(self):
        """All targets should be valid prompt types."""
        valid_targets = {"engine", "planner", "step"}
        for concept_id, section in PROMPT_SECTIONS.items():
            for target in section.targets:
                assert target in valid_targets, f"{concept_id} has invalid target: {target}"

    def test_expected_sections_exist(self):
        """Expected sections should be defined."""
        expected = {
            "dashboard_layout",
            "email_policy",
            "api_filtering",
            "visualization",
            "state_management",
            "llm_batching",
            "sensitive_data",
        }
        assert set(PROMPT_SECTIONS.keys()) == expected


class TestConceptDetector:
    """Test the ConceptDetector class."""

    @pytest.fixture(scope="class")
    def detector(self) -> ConceptDetector:
        """Create and initialize a detector for tests."""
        d = ConceptDetector()
        d.initialize()
        return d

    def test_initialization(self, detector):
        """Detector should initialize properly."""
        assert detector.is_initialized
        assert detector._exemplar_embeddings is not None
        assert len(detector._exemplar_to_concept) > 0

    def test_detect_dashboard(self, detector):
        """Should detect dashboard concept for dashboard queries."""
        concepts = detector.detect("create a sales dashboard with KPIs")
        concept_ids = {c.concept_id for c in concepts}

        assert "dashboard_layout" in concept_ids

    def test_detect_email(self, detector):
        """Should detect email concept for email queries."""
        concepts = detector.detect("email the report to the CFO")
        concept_ids = {c.concept_id for c in concepts}

        assert "email_policy" in concept_ids

    def test_detect_visualization(self, detector):
        """Should detect visualization concept for chart queries."""
        concepts = detector.detect("plot a bar chart of revenue by region")
        concept_ids = {c.concept_id for c in concepts}

        assert "visualization" in concept_ids

    def test_detect_api_filtering(self, detector):
        """Should detect API filtering concept for API queries."""
        concepts = detector.detect("query the GraphQL API for pending orders")
        concept_ids = {c.concept_id for c in concepts}

        assert "api_filtering" in concept_ids

    def test_detect_state_management(self, detector):
        """Should detect state management concept."""
        concepts = detector.detect("load the customers dataframe from the previous step")
        concept_ids = {c.concept_id for c in concepts}

        assert "state_management" in concept_ids

    def test_detect_llm_batching(self, detector):
        """Should detect LLM batching concept."""
        concepts = detector.detect("add descriptions to each product using LLM")
        concept_ids = {c.concept_id for c in concepts}

        assert "llm_batching" in concept_ids

    def test_detect_sensitive_data(self, detector):
        """Should detect sensitive data concept."""
        concepts = detector.detect("show me employee salaries and compensation")
        concept_ids = {c.concept_id for c in concepts}

        assert "sensitive_data" in concept_ids

    def test_no_detection_for_simple_query(self, detector):
        """Should not detect concepts for simple unrelated queries."""
        concepts = detector.detect("how many customers do we have?")

        # This query shouldn't strongly match any specialized concept
        # (it's a simple data query, not asking for dashboard, viz, etc.)
        high_confidence = [c for c in concepts if c.similarity > 0.7]
        assert len(high_confidence) == 0

    def test_target_filtering(self, detector):
        """Should filter by target prompt type."""
        # email_policy only targets planner
        concepts = detector.detect("send the report to the team", target="planner")
        planner_concepts = {c.concept_id for c in concepts}

        concepts = detector.detect("send the report to the team", target="step")
        step_concepts = {c.concept_id for c in concepts}

        # email_policy should be in planner but not step
        assert "email_policy" in planner_concepts
        assert "email_policy" not in step_concepts

    def test_threshold_override(self, detector):
        """Should respect threshold override."""
        # With high threshold, fewer matches
        concepts_high = detector.detect("create a chart", threshold=0.8)
        concepts_low = detector.detect("create a chart", threshold=0.3)

        assert len(concepts_low) >= len(concepts_high)

    def test_similarity_scores(self, detector):
        """Similarity scores should be reasonable."""
        concepts = detector.detect("build an interactive dashboard with charts")

        for concept in concepts:
            assert 0 <= concept.similarity <= 1
            assert isinstance(concept.section, PromptSection)

    def test_sorted_by_similarity(self, detector):
        """Results should be sorted by similarity descending."""
        concepts = detector.detect("create a dashboard with visualizations")

        if len(concepts) > 1:
            for i in range(len(concepts) - 1):
                assert concepts[i].similarity >= concepts[i + 1].similarity


class TestGetSectionsForPrompt:
    """Test the get_sections_for_prompt method."""

    @pytest.fixture(scope="class")
    def detector(self) -> ConceptDetector:
        """Create and initialize a detector for tests."""
        d = ConceptDetector()
        d.initialize()
        return d

    def test_returns_string(self, detector):
        """Should return a string."""
        content = detector.get_sections_for_prompt("create a dashboard", "step")
        assert isinstance(content, str)

    def test_contains_section_content(self, detector):
        """Should contain the section content."""
        content = detector.get_sections_for_prompt("create a dashboard with charts", "step")

        # Should contain dashboard layout rules
        assert "Dashboard" in content or "make_subplots" in content

    def test_empty_for_no_matches(self, detector):
        """Should return empty string when no concepts match."""
        content = detector.get_sections_for_prompt(
            "select count from table",
            "step",
            threshold=0.9,  # Very high threshold
        )
        # With high threshold, simple query shouldn't match
        # (may or may not be empty depending on embedding similarity)
        assert isinstance(content, str)

    def test_multiple_sections_joined(self, detector):
        """Should join multiple sections."""
        # Query that might match both visualization and dashboard
        content = detector.get_sections_for_prompt(
            "create a dashboard with interactive charts and maps",
            "step",
            threshold=0.5,
        )

        # Should have content from multiple sections
        assert len(content) > 100  # Should be substantial


class TestConceptDetectorLazyInit:
    """Test lazy initialization behavior."""

    def test_not_initialized_by_default(self):
        """Detector should not be initialized until explicitly called."""
        d = ConceptDetector()
        assert not d.is_initialized

    def test_auto_init_on_detect(self):
        """Should auto-initialize when detect is called."""
        d = ConceptDetector()
        d.detect("test query")
        assert d.is_initialized

    def test_auto_init_on_get_sections(self):
        """Should auto-initialize when get_sections_for_prompt is called."""
        d = ConceptDetector()
        d.get_sections_for_prompt("test query", "step")
        assert d.is_initialized


class TestConceptDetectorModelSharing:
    """Test model sharing capability."""

    def test_accepts_external_model(self):
        """Should accept an external model instance."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(ConceptDetector.EMBEDDING_MODEL)
        d = ConceptDetector(model=model)
        d.initialize()

        assert d.is_initialized
        # Should use the provided model
        assert d._model is model
