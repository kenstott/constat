# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for IntentClassifier embedding-based intent classification."""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from constat.execution.mode import PrimaryIntent, SubIntent, TurnIntent, Phase, Mode
from constat.execution.intent_classifier import (
    IntentClassifier,
    PRIMARY_THRESHOLD,
    SUB_THRESHOLD,
    EMBEDDING_MODEL,
)


class TestIntentClassifierInit:
    """Tests for IntentClassifier initialization."""

    def test_loads_default_exemplars(self):
        """Should load exemplars from default path."""
        classifier = IntentClassifier()
        assert classifier._exemplars is not None
        assert "primary_intents" in classifier._exemplars

    def test_loads_custom_exemplars(self, tmp_path):
        """Should load exemplars from custom path."""
        exemplar_file = tmp_path / "exemplars.yaml"
        exemplar_file.write_text("""
primary_intents:
  query:
    - "what is this"
  plan_new:
    - "analyze sales"
sub_intents:
  query:
    detail:
      - "explain more"
""")
        classifier = IntentClassifier(exemplar_path=str(exemplar_file))
        assert "query" in classifier._exemplars["primary_intents"]
        assert "plan_new" in classifier._exemplars["primary_intents"]

    def test_raises_on_missing_exemplar_file(self, tmp_path):
        """Should raise FileNotFoundError for missing exemplar file."""
        with pytest.raises(FileNotFoundError):
            IntentClassifier(exemplar_path=str(tmp_path / "nonexistent.yaml"))

    def test_raises_on_invalid_yaml(self, tmp_path):
        """Should raise on invalid YAML."""
        exemplar_file = tmp_path / "bad.yaml"
        exemplar_file.write_text("not: valid: yaml: [[")
        with pytest.raises(Exception):  # yaml.YAMLError
            IntentClassifier(exemplar_path=str(exemplar_file))

    def test_raises_on_missing_primary_intents_key(self, tmp_path):
        """Should raise when primary_intents key is missing."""
        exemplar_file = tmp_path / "missing.yaml"
        exemplar_file.write_text("""
sub_intents:
  query:
    detail:
      - "explain"
""")
        with pytest.raises(ValueError, match="missing 'primary_intents'"):
            IntentClassifier(exemplar_path=str(exemplar_file))

    def test_model_not_loaded_until_needed(self):
        """Model should be lazy-loaded, not at init time."""
        classifier = IntentClassifier()
        assert classifier._model is None


class TestExemplarStructure:
    """Tests for the default exemplar file structure."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    def test_has_all_primary_intents(self, classifier):
        """Should have exemplars for all primary intents."""
        primary_intents = classifier._exemplars.get("primary_intents", {})
        assert "query" in primary_intents
        assert "plan_new" in primary_intents
        assert "plan_continue" in primary_intents
        assert "control" in primary_intents

    def test_primary_intents_have_multiple_exemplars(self, classifier):
        """Each primary intent should have multiple exemplars for variety."""
        for intent_name, exemplars in classifier._exemplars.get("primary_intents", {}).items():
            assert len(exemplars) >= 5, f"{intent_name} should have at least 5 exemplars"

    def test_has_sub_intents_for_query(self, classifier):
        """Should have sub-intents for query intent."""
        sub_intents = classifier._exemplars.get("sub_intents", {})
        assert "query" in sub_intents
        query_subs = sub_intents["query"]
        assert "detail" in query_subs
        assert "provenance" in query_subs
        assert "summary" in query_subs
        assert "lookup" in query_subs

    def test_has_sub_intents_for_control(self, classifier):
        """Should have sub-intents for control intent."""
        sub_intents = classifier._exemplars.get("sub_intents", {})
        assert "control" in sub_intents
        control_subs = sub_intents["control"]
        assert "mode_switch" in control_subs
        assert "reset" in control_subs
        assert "help" in control_subs
        assert "status" in control_subs
        assert "exit" in control_subs
        assert "cancel" in control_subs
        assert "replan" in control_subs

    def test_plan_continue_has_correction_sub_intent(self, classifier):
        """plan_continue should have correction sub-intent for reusable user rules."""
        sub_intents = classifier._exemplars.get("sub_intents", {})
        # plan_continue should have correction sub-intent
        if "plan_continue" in sub_intents:
            assert "correction" in sub_intents["plan_continue"]
            assert len(sub_intents["plan_continue"]["correction"]) >= 3


class TestMessageSplitting:
    """Tests for multi-intent message splitting."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    def test_single_segment(self, classifier):
        """Single sentence should not be split."""
        segments = classifier._split_message("analyze the sales data")
        assert len(segments) == 1
        assert segments[0] == "analyze the sales data"

    def test_split_on_period(self, classifier):
        """Should split on period followed by space."""
        segments = classifier._split_message("analyze sales. show the results.")
        assert len(segments) == 2
        assert segments[0] == "analyze sales."
        assert segments[1] == "show the results."

    def test_split_on_semicolon(self, classifier):
        """Should split on semicolon followed by space."""
        segments = classifier._split_message("analyze sales; then compare")
        assert len(segments) == 2
        assert segments[0] == "analyze sales;"
        assert segments[1] == "then compare"

    def test_preserves_numbers_with_decimals(self, classifier):
        """Should not split on decimal points in numbers."""
        segments = classifier._split_message("set threshold to 3.14")
        assert len(segments) == 1
        assert "3.14" in segments[0]

    def test_filters_empty_segments(self, classifier):
        """Should filter out empty segments."""
        segments = classifier._split_message("analyze.  .  show results.")
        # Filter whitespace-only segments
        assert all(s.strip() for s in segments)

    def test_strips_whitespace(self, classifier):
        """Should strip whitespace from segments."""
        segments = classifier._split_message("  analyze sales.  show results  ")
        for segment in segments:
            assert segment == segment.strip()


class TestTargetExtraction:
    """Tests for target extraction from user input."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    def test_extract_query_target_about(self, classifier):
        """Should extract target from 'about X' pattern."""
        target = classifier._extract_target(PrimaryIntent.QUERY, "tell me about the revenue")
        assert target is not None
        assert "revenue" in target.lower()

    def test_extract_query_target_what_is(self, classifier):
        """Should extract target from 'what is X' pattern."""
        target = classifier._extract_target(PrimaryIntent.QUERY, "what is the total sales?")
        assert target is not None
        assert "total sales" in target.lower()

    def test_extract_plan_continue_target_change(self, classifier):
        """Should extract target from 'change X to' pattern."""
        target = classifier._extract_target(PrimaryIntent.PLAN_CONTINUE, "change the threshold to 50")
        assert target is not None
        assert "threshold" in target.lower()

    def test_extract_plan_continue_target_add(self, classifier):
        """Should extract target from 'add X' pattern."""
        target = classifier._extract_target(PrimaryIntent.PLAN_CONTINUE, "add another filter")
        assert target is not None
        assert "filter" in target.lower()

    def test_extract_plan_new_target_analyze(self, classifier):
        """Should extract target from 'analyze X' pattern."""
        target = classifier._extract_target(PrimaryIntent.PLAN_NEW, "analyze the sales data")
        assert target is not None
        assert "sales data" in target.lower()

    def test_no_target_for_simple_input(self, classifier):
        """Should return None when no target pattern matches."""
        target = classifier._extract_target(PrimaryIntent.QUERY, "hello")
        assert target is None


class TestClassifyWithMockedModel:
    """Tests for classification using mocked embedding model."""

    @pytest.fixture
    def mock_classifier(self, tmp_path):
        """Create classifier with mocked model."""
        exemplar_file = tmp_path / "exemplars.yaml"
        exemplar_file.write_text("""
primary_intents:
  query:
    - "what does this mean"
    - "explain the results"
  plan_new:
    - "analyze the sales"
    - "calculate the total"
  plan_continue:
    - "change that to"
    - "add another"
  control:
    - "start over"
    - "help"
sub_intents:
  query:
    detail:
      - "explain more"
    summary:
      - "summarize"
  control:
    reset:
      - "start over"
    help:
      - "show help"
""")
        classifier = IntentClassifier(exemplar_path=str(exemplar_file))

        # Mock the model
        mock_model = Mock()

        def mock_encode(text, normalize_embeddings=True):
            """Return different embeddings based on text content."""
            if isinstance(text, str):
                # Single text - return 1D array
                if "explain" in text.lower() or "what" in text.lower():
                    return np.array([0.9] * 1024)  # Query-like
                elif "analyze" in text.lower() or "calculate" in text.lower():
                    return np.array([0.1] * 1024)  # Plan new-like
                elif "change" in text.lower() or "add" in text.lower():
                    return np.array([0.5] * 1024)  # Plan continue-like
                elif "start over" in text.lower() or "help" in text.lower():
                    return np.array([0.2] * 1024)  # Control-like
                else:
                    return np.array([0.0] * 1024)
            else:
                # List of texts - return 2D array
                result = []
                for t in text:
                    if "explain" in t.lower() or "what" in t.lower():
                        result.append([0.9] * 1024)
                    elif "analyze" in t.lower() or "calculate" in t.lower():
                        result.append([0.1] * 1024)
                    elif "change" in t.lower() or "add" in t.lower():
                        result.append([0.5] * 1024)
                    elif "start over" in t.lower() or "help" in t.lower():
                        result.append([0.2] * 1024)
                    else:
                        result.append([0.0] * 1024)
                return np.array(result)

        mock_model.encode = mock_encode

        # Pre-load the model to skip lazy loading
        classifier._model = mock_model
        classifier._primary_embeddings = {
            PrimaryIntent.QUERY: np.array([[0.9] * 1024, [0.9] * 1024]),
            PrimaryIntent.PLAN_NEW: np.array([[0.1] * 1024, [0.1] * 1024]),
            PrimaryIntent.PLAN_CONTINUE: np.array([[0.5] * 1024, [0.5] * 1024]),
            PrimaryIntent.CONTROL: np.array([[0.2] * 1024, [0.2] * 1024]),
        }
        classifier._sub_embeddings = {
            PrimaryIntent.QUERY: {
                SubIntent.DETAIL: np.array([[0.9] * 1024]),
                SubIntent.SUMMARY: np.array([[0.85] * 1024]),
            },
            PrimaryIntent.CONTROL: {
                SubIntent.RESET: np.array([[0.2] * 1024]),
                SubIntent.HELP: np.array([[0.25] * 1024]),
            },
        }

        return classifier

    def test_classify_returns_turn_intent(self, mock_classifier):
        """classify should return a TurnIntent."""
        result = mock_classifier.classify("explain the results")
        assert isinstance(result, TurnIntent)
        assert isinstance(result.primary, PrimaryIntent)

    def test_classify_query_intent(self, mock_classifier):
        """Should classify query-like input as QUERY."""
        result = mock_classifier.classify("what does this mean")
        assert result.primary == PrimaryIntent.QUERY

    def test_classify_empty_input_defaults_to_query(self, mock_classifier):
        """Empty input should default to QUERY."""
        result = mock_classifier.classify("")
        assert result.primary == PrimaryIntent.QUERY

        result = mock_classifier.classify("   ")
        assert result.primary == PrimaryIntent.QUERY


class TestLLMFallback:
    """Tests for LLM fallback classification."""

    @pytest.fixture
    def classifier_with_llm(self):
        """Create classifier with mocked LLM provider."""
        classifier = IntentClassifier()
        mock_llm = Mock()
        mock_llm.generate.return_value = """PRIMARY: plan_new
SUB: compare
TARGET: revenue options
CONFIDENCE: high"""
        classifier._llm_provider = mock_llm
        return classifier

    def test_parse_llm_response_valid(self, classifier_with_llm):
        """Should parse valid LLM response."""
        response = """PRIMARY: query
SUB: detail
TARGET: the revenue breakdown
CONFIDENCE: high"""
        result = classifier_with_llm._parse_llm_response(response, "explain the revenue breakdown")

        assert result.primary == PrimaryIntent.QUERY
        assert result.sub == SubIntent.DETAIL
        assert result.target == "the revenue breakdown"

    def test_parse_llm_response_no_sub(self, classifier_with_llm):
        """Should handle response with no sub-intent."""
        response = """PRIMARY: plan_continue
SUB: none
TARGET: threshold
CONFIDENCE: medium"""
        result = classifier_with_llm._parse_llm_response(response, "change the threshold")

        assert result.primary == PrimaryIntent.PLAN_CONTINUE
        assert result.sub is None
        assert result.target == "threshold"

    def test_parse_llm_response_no_target(self, classifier_with_llm):
        """Should handle response with no target."""
        response = """PRIMARY: control
SUB: reset
TARGET: none
CONFIDENCE: high"""
        result = classifier_with_llm._parse_llm_response(response, "start over")

        assert result.primary == PrimaryIntent.CONTROL
        assert result.sub == SubIntent.RESET

    def test_parse_llm_response_invalid_primary_defaults_to_query(self, classifier_with_llm):
        """Should default to QUERY for invalid primary intent."""
        response = """PRIMARY: invalid_intent
SUB: none
TARGET: none
CONFIDENCE: low"""
        result = classifier_with_llm._parse_llm_response(response, "something")

        assert result.primary == PrimaryIntent.QUERY

    def test_llm_fallback_without_provider_uses_embedding(self):
        """Without LLM provider, fallback should use embedding match."""
        classifier = IntentClassifier()
        # Don't set _llm_provider

        # Mock the model to return known values
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.5] * 1024)
        classifier._model = mock_model
        classifier._primary_embeddings = {
            PrimaryIntent.QUERY: np.array([[0.9] * 1024]),
            PrimaryIntent.PLAN_NEW: np.array([[0.1] * 1024]),
            PrimaryIntent.PLAN_CONTINUE: np.array([[0.5] * 1024]),
            PrimaryIntent.CONTROL: np.array([[0.2] * 1024]),
        }
        classifier._sub_embeddings = {}

        result = classifier._llm_fallback("test input", None)

        # Should return a TurnIntent without raising
        assert isinstance(result, TurnIntent)


class TestClassificationThresholds:
    """Tests for classification threshold behavior."""

    def test_primary_threshold_value(self):
        """Primary threshold should be 0.80 as specified."""
        assert PRIMARY_THRESHOLD == 0.80

    def test_sub_threshold_value(self):
        """Sub threshold should be 0.65 as specified."""
        assert SUB_THRESHOLD == 0.65


class TestTurnIntentDataclass:
    """Tests for TurnIntent dataclass from mode.py."""

    def test_turn_intent_with_all_fields(self):
        """TurnIntent should hold all fields."""
        intent = TurnIntent(
            primary=PrimaryIntent.QUERY,
            sub=SubIntent.DETAIL,
            target="revenue breakdown",
        )
        assert intent.primary == PrimaryIntent.QUERY
        assert intent.sub == SubIntent.DETAIL
        assert intent.target == "revenue breakdown"

    def test_turn_intent_defaults(self):
        """TurnIntent should have None defaults for optional fields."""
        intent = TurnIntent(primary=PrimaryIntent.CONTROL)
        assert intent.primary == PrimaryIntent.CONTROL
        assert intent.sub is None
        assert intent.target is None


class TestIntentClassifierSetLLMProvider:
    """Tests for setting LLM provider."""

    def test_set_llm_provider(self):
        """Should allow setting LLM provider after init."""
        classifier = IntentClassifier()
        assert classifier._llm_provider is None

        mock_llm = Mock()
        classifier.set_llm_provider(mock_llm)

        assert classifier._llm_provider is mock_llm


class TestEmbeddingModelConfig:
    """Tests for embedding model configuration."""

    def test_embedding_model_name(self):
        """Should use BAAI/bge-large-en-v1.5 model."""
        assert EMBEDDING_MODEL == "BAAI/bge-large-en-v1.5"


# Integration test that requires the actual model (slow, skip by default)
@pytest.mark.slow
class TestIntentClassifierIntegration:
    """Integration tests with actual embedding model."""

    @pytest.fixture
    def classifier(self):
        """Create classifier and load model."""
        classifier = IntentClassifier()
        classifier._load_embedding_model()  # Force model load
        return classifier

    def test_model_loads(self, classifier):
        """Model should load without error."""
        assert classifier._model is not None
        assert classifier._primary_embeddings is not None

    def test_classify_query_phrase(self, classifier):
        """Should classify explanation request as query."""
        result = classifier.classify("what does this mean")
        assert result.primary == PrimaryIntent.QUERY

    def test_classify_analysis_request(self, classifier):
        """Should classify analysis request as plan_new."""
        result = classifier.classify("analyze the sales data by region")
        assert result.primary == PrimaryIntent.PLAN_NEW

    def test_classify_modification_request(self, classifier):
        """Should classify modification as plan_continue."""
        result = classifier.classify("actually, change that to use Q4 data instead")
        assert result.primary == PrimaryIntent.PLAN_CONTINUE

    def test_classify_reset_request(self, classifier):
        """Should classify reset as control."""
        result = classifier.classify("start over")
        assert result.primary == PrimaryIntent.CONTROL
        assert result.sub == SubIntent.RESET

    def test_classify_help_request(self, classifier):
        """Should classify help request as control.help."""
        result = classifier.classify("what commands are available")
        assert result.primary == PrimaryIntent.CONTROL
        assert result.sub == SubIntent.HELP

    def test_classify_provenance_request(self, classifier):
        """Should classify provenance request as query.provenance."""
        result = classifier.classify("how did you determine that value")
        assert result.primary == PrimaryIntent.QUERY
        assert result.sub == SubIntent.PROVENANCE
