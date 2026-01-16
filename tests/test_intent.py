"""Tests for intent classification module."""

import pytest

from constat.execution.intent import (
    FollowUpIntent,
    DetectedIntent,
    IntentClassification,
    IMPLIES_REDO,
    QUICK_INTENTS,
    EXECUTION_INTENTS,
    INTENT_SUGGESTED_ACTIONS,
    ORDER_CONFIRMATION_THRESHOLD,
    from_analysis,
)


class TestFollowUpIntent:
    """Tests for the FollowUpIntent enum."""

    def test_all_intents_defined(self):
        """Verify all expected intents are defined."""
        expected = {
            "REDO", "MODIFY_FACT", "STEER_PLAN", "DRILL_DOWN", "REFINE_SCOPE",
            "CHALLENGE", "EXPORT", "EXTEND", "MODE_SWITCH", "PROVENANCE",
            "CREATE_ARTIFACT", "NEW_QUESTION", "TRIGGER_ACTION", "COMPARE",
            "PREDICT", "LOOKUP", "ALERT", "SUMMARIZE", "QUERY", "RESET",
        }
        actual = {intent.name for intent in FollowUpIntent}
        assert actual == expected

    def test_intent_values_are_lowercase(self):
        """Intent values should be lowercase snake_case."""
        for intent in FollowUpIntent:
            assert intent.value == intent.value.lower()
            assert intent.value == intent.name.lower()


class TestIntentConstants:
    """Tests for intent category constants."""

    def test_implies_redo_contains_expected(self):
        """IMPLIES_REDO should contain intents that logically require re-execution."""
        assert FollowUpIntent.MODIFY_FACT in IMPLIES_REDO
        assert FollowUpIntent.REFINE_SCOPE in IMPLIES_REDO
        assert FollowUpIntent.STEER_PLAN in IMPLIES_REDO
        # REDO itself doesn't imply redo - it IS redo
        assert FollowUpIntent.REDO not in IMPLIES_REDO

    def test_quick_intents_are_cheap(self):
        """QUICK_INTENTS should be cheap/fast operations."""
        assert FollowUpIntent.LOOKUP in QUICK_INTENTS
        assert FollowUpIntent.PROVENANCE in QUICK_INTENTS
        assert FollowUpIntent.EXPORT in QUICK_INTENTS
        assert FollowUpIntent.DRILL_DOWN in QUICK_INTENTS
        # Expensive operations should not be quick
        assert FollowUpIntent.REDO not in QUICK_INTENTS
        assert FollowUpIntent.COMPARE not in QUICK_INTENTS

    def test_execution_intents_are_expensive(self):
        """EXECUTION_INTENTS should be potentially expensive operations."""
        assert FollowUpIntent.REDO in EXECUTION_INTENTS
        assert FollowUpIntent.MODIFY_FACT in EXECUTION_INTENTS
        assert FollowUpIntent.COMPARE in EXECUTION_INTENTS
        assert FollowUpIntent.PREDICT in EXECUTION_INTENTS
        # Quick operations should not be execution intents
        assert FollowUpIntent.LOOKUP not in EXECUTION_INTENTS
        assert FollowUpIntent.EXPORT not in EXECUTION_INTENTS

    def test_order_confirmation_threshold_is_reasonable(self):
        """ORDER_CONFIRMATION_THRESHOLD should be a reasonable value."""
        assert ORDER_CONFIRMATION_THRESHOLD >= 2
        assert ORDER_CONFIRMATION_THRESHOLD <= 5


class TestDetectedIntent:
    """Tests for the DetectedIntent dataclass."""

    def test_default_confidence(self):
        """Default confidence should be 0.8."""
        detected = DetectedIntent(intent=FollowUpIntent.REDO)
        assert detected.confidence == 0.8

    def test_with_extracted_value(self):
        """Should store extracted value."""
        detected = DetectedIntent(
            intent=FollowUpIntent.MODIFY_FACT,
            confidence=0.9,
            extracted_value="threshold=50k",
        )
        assert detected.intent == FollowUpIntent.MODIFY_FACT
        assert detected.confidence == 0.9
        assert detected.extracted_value == "threshold=50k"


class TestIntentClassification:
    """Tests for the IntentClassification dataclass."""

    def test_primary_intent_with_single(self):
        """primary_intent should return the first intent."""
        classification = IntentClassification(
            intents=[DetectedIntent(intent=FollowUpIntent.REDO)],
            original_text="redo the analysis",
        )
        assert classification.primary_intent == FollowUpIntent.REDO

    def test_primary_intent_with_multiple(self):
        """primary_intent should return the first of multiple intents."""
        classification = IntentClassification(
            intents=[
                DetectedIntent(intent=FollowUpIntent.MODIFY_FACT),
                DetectedIntent(intent=FollowUpIntent.REDO),
            ],
            original_text="change threshold and redo",
        )
        assert classification.primary_intent == FollowUpIntent.MODIFY_FACT

    def test_primary_intent_empty(self):
        """primary_intent should return None when no intents."""
        classification = IntentClassification(intents=[], original_text="")
        assert classification.primary_intent is None

    def test_has_multiple_intents(self):
        """has_multiple_intents should detect multiple intents."""
        single = IntentClassification(
            intents=[DetectedIntent(intent=FollowUpIntent.REDO)],
            original_text="redo",
        )
        assert not single.has_multiple_intents

        multiple = IntentClassification(
            intents=[
                DetectedIntent(intent=FollowUpIntent.MODIFY_FACT),
                DetectedIntent(intent=FollowUpIntent.REDO),
            ],
            original_text="change and redo",
        )
        assert multiple.has_multiple_intents

    def test_has_intent(self):
        """has_intent should check for specific intents."""
        classification = IntentClassification(
            intents=[
                DetectedIntent(intent=FollowUpIntent.MODIFY_FACT),
                DetectedIntent(intent=FollowUpIntent.REDO),
            ],
            original_text="change and redo",
        )
        assert classification.has_intent(FollowUpIntent.MODIFY_FACT)
        assert classification.has_intent(FollowUpIntent.REDO)
        assert not classification.has_intent(FollowUpIntent.EXPORT)

    def test_get_intent_confidence(self):
        """get_intent_confidence should return confidence for specific intent."""
        classification = IntentClassification(
            intents=[
                DetectedIntent(intent=FollowUpIntent.MODIFY_FACT, confidence=0.95),
                DetectedIntent(intent=FollowUpIntent.REDO, confidence=0.85),
            ],
            original_text="test",
        )
        assert classification.get_intent_confidence(FollowUpIntent.MODIFY_FACT) == 0.95
        assert classification.get_intent_confidence(FollowUpIntent.REDO) == 0.85
        assert classification.get_intent_confidence(FollowUpIntent.EXPORT) == 0.0

    def test_implies_redo(self):
        """implies_redo should detect intents that imply re-execution."""
        # MODIFY_FACT implies redo
        modify = IntentClassification(
            intents=[DetectedIntent(intent=FollowUpIntent.MODIFY_FACT)],
            original_text="change threshold",
        )
        assert modify.implies_redo

        # EXPORT does not imply redo
        export = IntentClassification(
            intents=[DetectedIntent(intent=FollowUpIntent.EXPORT)],
            original_text="export results",
        )
        assert not export.implies_redo

    def test_requires_execution(self):
        """requires_execution should detect expensive intents."""
        # REDO requires execution
        redo = IntentClassification(
            intents=[DetectedIntent(intent=FollowUpIntent.REDO)],
            original_text="redo",
        )
        assert redo.requires_execution

        # LOOKUP does not require execution
        lookup = IntentClassification(
            intents=[DetectedIntent(intent=FollowUpIntent.LOOKUP)],
            original_text="who owns X?",
        )
        assert not lookup.requires_execution

    def test_is_quick(self):
        """is_quick should detect quick/cheap intents."""
        # LOOKUP is quick
        lookup = IntentClassification(
            intents=[DetectedIntent(intent=FollowUpIntent.LOOKUP)],
            original_text="status?",
        )
        assert lookup.is_quick

        # REDO is not quick
        redo = IntentClassification(
            intents=[DetectedIntent(intent=FollowUpIntent.REDO)],
            original_text="redo",
        )
        assert not redo.is_quick

        # Mixed: if any is not quick, result is not quick
        mixed = IntentClassification(
            intents=[
                DetectedIntent(intent=FollowUpIntent.LOOKUP),
                DetectedIntent(intent=FollowUpIntent.REDO),
            ],
            original_text="lookup and redo",
        )
        assert not mixed.is_quick


class TestIntentOrderConfirmation:
    """Tests for intent order confirmation features."""

    def test_should_confirm_order_below_threshold(self):
        """should_confirm_order should be False for few intents."""
        # 1 intent
        one = IntentClassification(
            intents=[DetectedIntent(intent=FollowUpIntent.REDO)],
            original_text="redo",
        )
        assert not one.should_confirm_order

        # 2 intents
        two = IntentClassification(
            intents=[
                DetectedIntent(intent=FollowUpIntent.MODIFY_FACT),
                DetectedIntent(intent=FollowUpIntent.REDO),
            ],
            original_text="change and redo",
        )
        assert not two.should_confirm_order

    def test_should_confirm_order_at_threshold(self):
        """should_confirm_order should be True at threshold."""
        intents = [
            DetectedIntent(intent=FollowUpIntent.MODIFY_FACT, extracted_value="threshold=50k"),
            DetectedIntent(intent=FollowUpIntent.REFINE_SCOPE, extracted_value="California"),
            DetectedIntent(intent=FollowUpIntent.REDO),
        ]
        classification = IntentClassification(intents=intents, original_text="test")
        assert len(intents) == ORDER_CONFIRMATION_THRESHOLD
        assert classification.should_confirm_order

    def test_should_confirm_order_above_threshold(self):
        """should_confirm_order should be True above threshold."""
        intents = [
            DetectedIntent(intent=FollowUpIntent.MODIFY_FACT),
            DetectedIntent(intent=FollowUpIntent.REFINE_SCOPE),
            DetectedIntent(intent=FollowUpIntent.STEER_PLAN),
            DetectedIntent(intent=FollowUpIntent.REDO),
        ]
        classification = IntentClassification(intents=intents, original_text="test")
        assert len(intents) > ORDER_CONFIRMATION_THRESHOLD
        assert classification.should_confirm_order

    def test_get_order_confirmation_prompt_empty(self):
        """Confirmation prompt for empty intents."""
        classification = IntentClassification(intents=[], original_text="")
        prompt = classification.get_order_confirmation_prompt()
        assert prompt == "No intents detected."

    def test_get_order_confirmation_prompt_formatting(self):
        """Confirmation prompt should be well-formatted."""
        intents = [
            DetectedIntent(intent=FollowUpIntent.MODIFY_FACT, extracted_value="threshold=50k"),
            DetectedIntent(intent=FollowUpIntent.REFINE_SCOPE, extracted_value="California only"),
            DetectedIntent(intent=FollowUpIntent.REDO),
        ]
        classification = IntentClassification(intents=intents, original_text="test")
        prompt = classification.get_order_confirmation_prompt()

        # Should contain header
        assert "I detected these actions in the following order:" in prompt
        # Should contain numbered items
        assert "1. Modify Fact: threshold=50k" in prompt
        assert "2. Refine Scope: California only" in prompt
        assert "3. Redo" in prompt
        # Should contain confirmation question
        assert "Is this the correct order?" in prompt

    def test_get_order_confirmation_prompt_without_extracted_values(self):
        """Confirmation prompt should work without extracted values."""
        intents = [
            DetectedIntent(intent=FollowUpIntent.REDO),
            DetectedIntent(intent=FollowUpIntent.EXPORT),
            DetectedIntent(intent=FollowUpIntent.DRILL_DOWN),
        ]
        classification = IntentClassification(intents=intents, original_text="test")
        prompt = classification.get_order_confirmation_prompt()

        # Should not have colons after intent names (no extracted values)
        assert "1. Redo\n" in prompt or "1. Redo" in prompt
        assert "2. Export\n" in prompt or "2. Export" in prompt
        assert "3. Drill Down\n" in prompt or "3. Drill Down" in prompt


class TestIntentReordering:
    """Tests for intent reordering functionality."""

    def test_reorder_intents_basic(self):
        """reorder_intents should reorder correctly."""
        original = IntentClassification(
            intents=[
                DetectedIntent(intent=FollowUpIntent.REDO),
                DetectedIntent(intent=FollowUpIntent.MODIFY_FACT),
                DetectedIntent(intent=FollowUpIntent.EXPORT),
            ],
            original_text="test",
            fact_modifications=[{"fact": "test"}],
            scope_refinements=["California"],
        )

        # Move EXPORT to first, then REDO, then MODIFY_FACT
        reordered = original.reorder_intents([2, 0, 1])

        assert reordered.intents[0].intent == FollowUpIntent.EXPORT
        assert reordered.intents[1].intent == FollowUpIntent.REDO
        assert reordered.intents[2].intent == FollowUpIntent.MODIFY_FACT

    def test_reorder_intents_preserves_metadata(self):
        """reorder_intents should preserve other classification data."""
        original = IntentClassification(
            intents=[
                DetectedIntent(intent=FollowUpIntent.REDO),
                DetectedIntent(intent=FollowUpIntent.MODIFY_FACT),
            ],
            original_text="original text",
            fact_modifications=[{"fact_name": "threshold", "new_value": "50k"}],
            scope_refinements=["California"],
            target_mode="auditable",
        )

        reordered = original.reorder_intents([1, 0])

        assert reordered.original_text == "original text"
        assert reordered.fact_modifications == [{"fact_name": "threshold", "new_value": "50k"}]
        assert reordered.scope_refinements == ["California"]
        assert reordered.target_mode == "auditable"

    def test_reorder_intents_invalid_length(self):
        """reorder_intents should raise error for wrong length."""
        classification = IntentClassification(
            intents=[
                DetectedIntent(intent=FollowUpIntent.REDO),
                DetectedIntent(intent=FollowUpIntent.MODIFY_FACT),
            ],
            original_text="test",
        )

        with pytest.raises(ValueError, match="must have 2 elements"):
            classification.reorder_intents([0, 1, 2])  # Too many

        with pytest.raises(ValueError, match="must have 2 elements"):
            classification.reorder_intents([0])  # Too few

    def test_reorder_intents_invalid_indices(self):
        """reorder_intents should raise error for invalid indices."""
        classification = IntentClassification(
            intents=[
                DetectedIntent(intent=FollowUpIntent.REDO),
                DetectedIntent(intent=FollowUpIntent.MODIFY_FACT),
            ],
            original_text="test",
        )

        with pytest.raises(ValueError, match="must be a permutation"):
            classification.reorder_intents([0, 0])  # Duplicate index

        with pytest.raises(ValueError, match="must be a permutation"):
            classification.reorder_intents([0, 5])  # Out of range

    def test_reorder_intents_identity(self):
        """reorder_intents with identity permutation should preserve order."""
        original = IntentClassification(
            intents=[
                DetectedIntent(intent=FollowUpIntent.REDO),
                DetectedIntent(intent=FollowUpIntent.MODIFY_FACT),
                DetectedIntent(intent=FollowUpIntent.EXPORT),
            ],
            original_text="test",
        )

        reordered = original.reorder_intents([0, 1, 2])

        for i, intent in enumerate(original.intents):
            assert reordered.intents[i].intent == intent.intent


class TestAutoExecuteAndSuggestions:
    """Tests for auto-execution and suggestion logic."""

    def test_should_auto_execute_quick_intents(self):
        """Quick intents should always auto-execute."""
        lookup = IntentClassification(
            intents=[DetectedIntent(intent=FollowUpIntent.LOOKUP)],
            original_text="test",
        )
        assert lookup.should_auto_execute(expensive=False)
        assert lookup.should_auto_execute(expensive=True)

    def test_should_auto_execute_redo(self):
        """Explicit REDO auto-executes unless previous was expensive."""
        redo = IntentClassification(
            intents=[DetectedIntent(intent=FollowUpIntent.REDO)],
            original_text="test",
        )
        assert redo.should_auto_execute(expensive=False)
        # REDO is in EXECUTION_INTENTS, so it doesn't auto-execute when expensive
        assert not redo.should_auto_execute(expensive=True)

    def test_should_auto_execute_expensive_operation(self):
        """Expensive execution intents should not auto-execute when previous was expensive."""
        compare = IntentClassification(
            intents=[DetectedIntent(intent=FollowUpIntent.COMPARE)],
            original_text="test",
        )
        assert compare.should_auto_execute(expensive=False)
        assert not compare.should_auto_execute(expensive=True)

    def test_get_execution_plan_structure(self):
        """get_execution_plan should return proper structure."""
        classification = IntentClassification(
            intents=[
                DetectedIntent(intent=FollowUpIntent.MODIFY_FACT),
                DetectedIntent(intent=FollowUpIntent.REDO),
            ],
            original_text="test",
        )

        plan = classification.get_execution_plan(expensive=False)

        assert "auto_execute" in plan
        assert "suggest" in plan
        assert "requires_confirmation" in plan
        assert isinstance(plan["auto_execute"], list)
        assert isinstance(plan["suggest"], list)
        assert isinstance(plan["requires_confirmation"], bool)


class TestFromAnalysis:
    """Tests for the from_analysis converter function."""

    def test_from_analysis_with_intents(self):
        """from_analysis should convert QuestionAnalysis-like object."""

        class MockAnalysis:
            intents = [
                type("DetectedIntent", (), {"intent": "REDO", "confidence": 0.9, "extracted_value": None})(),
                type("DetectedIntent", (), {"intent": "MODIFY_FACT", "confidence": 0.8, "extracted_value": "x=5"})(),
            ]
            fact_modifications = [{"fact_name": "x", "new_value": "5"}]
            scope_refinements = ["Q4"]

        result = from_analysis(MockAnalysis())

        assert len(result.intents) == 2
        assert result.intents[0].intent == FollowUpIntent.REDO
        assert result.intents[0].confidence == 0.9
        assert result.intents[1].intent == FollowUpIntent.MODIFY_FACT
        assert result.intents[1].extracted_value == "x=5"
        assert result.fact_modifications == [{"fact_name": "x", "new_value": "5"}]
        assert result.scope_refinements == ["Q4"]

    def test_from_analysis_empty_intents(self):
        """from_analysis should default to NEW_QUESTION when no intents."""

        class MockAnalysis:
            intents = []
            fact_modifications = []
            scope_refinements = []

        result = from_analysis(MockAnalysis())

        assert len(result.intents) == 1
        assert result.intents[0].intent == FollowUpIntent.NEW_QUESTION
        assert result.intents[0].confidence == 0.5

    def test_from_analysis_invalid_intent(self):
        """from_analysis should skip invalid intent names."""

        class MockAnalysis:
            intents = [
                type("DetectedIntent", (), {"intent": "INVALID_INTENT", "confidence": 0.9, "extracted_value": None})(),
                type("DetectedIntent", (), {"intent": "REDO", "confidence": 0.8, "extracted_value": None})(),
            ]
            fact_modifications = []
            scope_refinements = []

        result = from_analysis(MockAnalysis())

        # Invalid intent should be skipped
        assert len(result.intents) == 1
        assert result.intents[0].intent == FollowUpIntent.REDO