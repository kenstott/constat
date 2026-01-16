"""Intent classification for interactive sessions.

Intent classification is performed by the LLM during question analysis
(in session._analyze_question), not through keyword matching. This module
provides the intent enum and helper classes/constants for intent handling.

A single prompt can express multiple intents (e.g., "change this assumption and redo").
The LLM preserves the natural order of intents for sequential execution.

Mode Preservation:
When a follow-up question contains a redo-like intent (REDO, PREDICT, MODIFY_FACT,
REFINE_SCOPE, or STEER_PLAN), the previous execution mode is preserved. This ensures
that "what if" questions and fact modifications stay in the same mode as the original
analysis. An explicit MODE_SWITCH intent overrides this behavior.

Intent Categories:
- REDO: Re-run the previous analysis
- MODIFY_FACT: Change, add, or remove a fact/assumption
- STEER_PLAN: Modify the execution plan (skip steps, add steps, change approach)
- DRILL_DOWN: Expand or explain a specific conclusion
- REFINE_SCOPE: Filter or narrow the analysis
- CHALLENGE: Verify or question a conclusion
- EXPORT: Save or export results
- EXTEND: Continue with additional analysis
- MODE_SWITCH: Change execution mode (overrides mode preservation)
- PROVENANCE: Show derivation or audit trail
- CREATE_ARTIFACT: Create a new artifact (dashboard, document, file)
- NEW_QUESTION: A completely new, unrelated question
- TRIGGER_ACTION: Execute a workflow or action (send email, schedule, etc.)
- COMPARE: Compare entities, periods, or scenarios
- PREDICT: What-if analysis or forecasting
- LOOKUP: Simple fact lookup
- ALERT: Set up monitoring or alerts
- SUMMARIZE: Summarize or condense results
- QUERY: Run a direct SQL query
- RESET: Clear session and start fresh
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FollowUpIntent(Enum):
    """Categories of user intent in prompts."""

    REDO = "redo"
    """Re-run the previous analysis, possibly with modifications."""

    MODIFY_FACT = "modify_fact"
    """Change, add, or remove a fact or assumption."""

    STEER_PLAN = "steer_plan"
    """Modify the execution plan structure."""

    DRILL_DOWN = "drill_down"
    """Expand or explain a specific conclusion."""

    REFINE_SCOPE = "refine_scope"
    """Filter or narrow the analysis scope."""

    CHALLENGE = "challenge"
    """Verify or question a conclusion."""

    EXPORT = "export"
    """Save or export results."""

    EXTEND = "extend"
    """Continue with additional related analysis."""

    MODE_SWITCH = "mode_switch"
    """Change execution mode."""

    PROVENANCE = "provenance"
    """Show derivation or audit trail."""

    CREATE_ARTIFACT = "create_artifact"
    """Create a new artifact (dashboard, document, file)."""

    NEW_QUESTION = "new_question"
    """A completely new, unrelated question."""

    TRIGGER_ACTION = "trigger_action"
    """Execute a workflow or action."""

    COMPARE = "compare"
    """Compare entities, periods, or scenarios."""

    PREDICT = "predict"
    """What-if analysis or forecasting."""

    LOOKUP = "lookup"
    """Simple fact lookup."""

    ALERT = "alert"
    """Set up monitoring or alerts."""

    SUMMARIZE = "summarize"
    """Summarize or condense results, provide a high-level overview."""

    QUERY = "query"
    """Run a direct SQL query or data retrieval."""

    RESET = "reset"
    """Clear session state and start fresh."""


# Intents that imply the user wants to re-run the analysis with changes
IMPLIES_REDO: set[FollowUpIntent] = {
    FollowUpIntent.MODIFY_FACT,
    FollowUpIntent.REFINE_SCOPE,
    FollowUpIntent.STEER_PLAN,
}


# Suggested actions based on detected intent (when not auto-executing)
INTENT_SUGGESTED_ACTIONS: dict[FollowUpIntent, tuple[str, str]] = {
    FollowUpIntent.MODIFY_FACT: ("Redo with updated values", "redo"),
    FollowUpIntent.REFINE_SCOPE: ("Rerun for narrowed scope", "redo"),
    FollowUpIntent.STEER_PLAN: ("Run the modified plan", "redo"),
    FollowUpIntent.REDO: ("Rerun the analysis", "redo"),
    FollowUpIntent.EXPORT: ("Export results", "export"),
    FollowUpIntent.CREATE_ARTIFACT: ("Create the artifact", "create"),
    FollowUpIntent.TRIGGER_ACTION: ("Execute the action", "trigger"),
    FollowUpIntent.COMPARE: ("Run comparison analysis", "compare"),
    FollowUpIntent.PREDICT: ("Run prediction/forecast", "predict"),
    FollowUpIntent.ALERT: ("Set up the alert", "alert"),
    FollowUpIntent.DRILL_DOWN: ("Show detailed breakdown", "drill_down"),
    FollowUpIntent.PROVENANCE: ("Show derivation chain", "provenance"),
    FollowUpIntent.CHALLENGE: ("Verify the conclusion", "challenge"),
    FollowUpIntent.LOOKUP: ("Look up the information", "lookup"),
    FollowUpIntent.SUMMARIZE: ("Summarize the results", "summarize"),
    FollowUpIntent.QUERY: ("Run the query", "query"),
    FollowUpIntent.RESET: ("Clear and start fresh", "reset"),
}


# Intents that are typically quick/cheap to execute
QUICK_INTENTS: set[FollowUpIntent] = {
    FollowUpIntent.LOOKUP,
    FollowUpIntent.PROVENANCE,
    FollowUpIntent.EXPORT,
    FollowUpIntent.DRILL_DOWN,
    FollowUpIntent.SUMMARIZE,
    FollowUpIntent.QUERY,
    FollowUpIntent.RESET,
}


# Intents that typically require re-execution (potentially expensive)
EXECUTION_INTENTS: set[FollowUpIntent] = {
    FollowUpIntent.REDO,
    FollowUpIntent.MODIFY_FACT,
    FollowUpIntent.REFINE_SCOPE,
    FollowUpIntent.STEER_PLAN,
    FollowUpIntent.COMPARE,
    FollowUpIntent.PREDICT,
    FollowUpIntent.CREATE_ARTIFACT,
    FollowUpIntent.NEW_QUESTION,
    FollowUpIntent.ALERT,
}


# Threshold for recommending order confirmation with user
# When intent count >= this, suggest confirming execution order
ORDER_CONFIRMATION_THRESHOLD: int = 3


@dataclass
class DetectedIntent:
    """A single detected intent with context."""
    intent: FollowUpIntent
    confidence: float = 0.8
    extracted_value: Optional[str] = None


@dataclass
class IntentClassification:
    """Result of classifying a prompt.

    A prompt can have multiple intents, ranked by confidence.
    Intents are preserved in the order the user expressed them.
    """
    intents: list[DetectedIntent]
    original_text: str

    # Extracted context
    fact_modifications: list[dict] = field(default_factory=list)
    scope_refinements: list[str] = field(default_factory=list)
    target_mode: Optional[str] = None

    # Output preferences (detected by LLM, not keywords)
    wants_brief: bool = False
    """User wants brief/concise output without detailed insights."""

    @property
    def primary_intent(self) -> Optional[FollowUpIntent]:
        """Get the first intent (typically the main action)."""
        if not self.intents:
            return None
        return self.intents[0].intent

    @property
    def has_multiple_intents(self) -> bool:
        """Check if multiple intents were detected."""
        return len(self.intents) > 1

    def has_intent(self, intent: FollowUpIntent) -> bool:
        """Check if a specific intent was detected."""
        return any(d.intent == intent for d in self.intents)

    def get_intent_confidence(self, intent: FollowUpIntent) -> float:
        """Get confidence for a specific intent (0 if not detected)."""
        for d in self.intents:
            if d.intent == intent:
                return d.confidence
        return 0.0

    @property
    def implies_redo(self) -> bool:
        """Check if detected intents imply the user wants to redo the analysis."""
        return any(d.intent in IMPLIES_REDO for d in self.intents)

    @property
    def requires_execution(self) -> bool:
        """Check if detected intents require execution (potentially expensive)."""
        return any(d.intent in EXECUTION_INTENTS for d in self.intents)

    @property
    def is_quick(self) -> bool:
        """Check if all detected intents are quick/cheap to execute."""
        if not self.intents:
            return True
        return all(d.intent in QUICK_INTENTS for d in self.intents)

    @property
    def should_confirm_order(self) -> bool:
        """Check if the execution order should be confirmed with the user.

        Recommended when there are many intents, as the inferred order
        may not match user expectations (especially with complex temporal
        or priority words like 'before', 'after', 'always', 'first').
        """
        return len(self.intents) >= ORDER_CONFIRMATION_THRESHOLD

    def get_order_confirmation_prompt(self) -> str:
        """Get a formatted prompt for confirming intent execution order.

        Returns:
            A human-readable string showing the planned execution order.
        """
        if not self.intents:
            return "No intents detected."

        lines = ["I detected these actions in the following order:"]
        for i, detected in enumerate(self.intents, 1):
            intent_name = detected.intent.value.replace("_", " ").title()
            if detected.extracted_value:
                lines.append(f"  {i}. {intent_name}: {detected.extracted_value}")
            else:
                lines.append(f"  {i}. {intent_name}")

        lines.append("\nIs this the correct order? (yes/no/reorder)")
        return "\n".join(lines)

    def reorder_intents(self, new_order: list[int]) -> "IntentClassification":
        """Create a new IntentClassification with reordered intents.

        Args:
            new_order: List of 0-based indices specifying the new order.
                       e.g., [2, 0, 1] moves the third intent to first.

        Returns:
            A new IntentClassification with reordered intents.
        """
        if len(new_order) != len(self.intents):
            raise ValueError(f"new_order must have {len(self.intents)} elements")
        if set(new_order) != set(range(len(self.intents))):
            raise ValueError("new_order must be a permutation of intent indices")

        reordered = [self.intents[i] for i in new_order]
        return IntentClassification(
            intents=reordered,
            original_text=self.original_text,
            fact_modifications=self.fact_modifications,
            scope_refinements=self.scope_refinements,
            target_mode=self.target_mode,
            wants_brief=self.wants_brief,
        )

    def should_auto_execute(self, expensive: bool = False) -> bool:
        """Determine if the intent should be auto-executed or suggested.

        Args:
            expensive: True if the previous operation was expensive

        Returns:
            True if the intent should be auto-executed,
            False if it should be offered as a suggestion
        """
        if self.is_quick:
            return True
        if expensive and self.requires_execution:
            return False
        if self.has_intent(FollowUpIntent.REDO):
            return True
        if self.implies_redo:
            return not expensive
        return True

    def get_suggested_actions(
        self,
        expensive: bool = False,
        include_auto: bool = False,
    ) -> list[dict]:
        """Get suggested follow-up actions based on detected intents."""
        suggestions = []

        for detected in self.intents:
            intent = detected.intent
            if intent not in INTENT_SUGGESTED_ACTIONS:
                continue

            action_desc, verb = INTENT_SUGGESTED_ACTIONS[intent]
            would_auto = self._would_auto_execute_intent(intent, expensive)

            if include_auto or not would_auto:
                if intent == FollowUpIntent.MODIFY_FACT and self.fact_modifications:
                    mods = self.fact_modifications
                    if len(mods) == 1:
                        action_desc = f"Redo with {mods[0].get('fact_name', 'value')}={mods[0].get('new_value', '?')}"
                    else:
                        action_desc = f"Redo with {len(mods)} updated values"

                if intent == FollowUpIntent.REFINE_SCOPE and self.scope_refinements:
                    scope = self.scope_refinements[0]
                    action_desc = f"Rerun filtered to {scope}"

                suggestions.append({
                    "action": action_desc,
                    "verb": verb,
                    "intent": intent,
                    "auto": would_auto,
                })

        return suggestions

    def _would_auto_execute_intent(self, intent: FollowUpIntent, expensive: bool) -> bool:
        """Check if a specific intent would be auto-executed."""
        if intent in QUICK_INTENTS:
            return True
        if expensive and intent in EXECUTION_INTENTS:
            return False
        if intent == FollowUpIntent.REDO:
            return True
        if intent in IMPLIES_REDO:
            return not expensive
        return True

    def get_execution_plan(self, expensive: bool = False) -> dict:
        """Get a structured execution plan for handling the detected intents."""
        auto_execute = []
        suggest = []

        for detected in self.intents:
            intent = detected.intent
            if self._would_auto_execute_intent(intent, expensive):
                auto_execute.append(intent)
            elif intent in INTENT_SUGGESTED_ACTIONS:
                suggest.append(INTENT_SUGGESTED_ACTIONS[intent])

        requires_confirmation = (
            expensive and
            self.requires_execution and
            not self.has_intent(FollowUpIntent.REDO)
        )

        return {
            "auto_execute": auto_execute,
            "suggest": suggest,
            "requires_confirmation": requires_confirmation,
        }


def from_analysis(analysis) -> IntentClassification:
    """Convert a QuestionAnalysis to an IntentClassification.

    This bridges the session's analysis output to the intent module's format.

    Args:
        analysis: A QuestionAnalysis object from session._analyze_question

    Returns:
        IntentClassification with the detected intents
    """
    intents = []
    for detected in getattr(analysis, 'intents', []):
        # Handle both DetectedIntent objects and dicts
        if hasattr(detected, 'intent'):
            try:
                intent_enum = FollowUpIntent[detected.intent.upper()] if isinstance(detected.intent, str) else detected.intent
                intents.append(DetectedIntent(
                    intent=intent_enum,
                    confidence=getattr(detected, 'confidence', 0.8),
                    extracted_value=getattr(detected, 'extracted_value', None),
                ))
            except (KeyError, AttributeError):
                pass

    if not intents:
        intents.append(DetectedIntent(intent=FollowUpIntent.NEW_QUESTION, confidence=0.5))

    return IntentClassification(
        intents=intents,
        original_text="",
        fact_modifications=getattr(analysis, 'fact_modifications', []),
        scope_refinements=getattr(analysis, 'scope_refinements', []),
        wants_brief=getattr(analysis, 'wants_brief', False),
    )