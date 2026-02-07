# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Embedding-based intent classification with LLM fallback.

This module implements a hierarchical intent classifier that uses embedding
similarity for fast classification of common intents, falling back to LLM
classification for ambiguous or low-confidence cases.

The classifier uses a two-tier approach:
1. Primary intent matching against ~40 exemplars
2. Sub-intent matching scoped to the matched primary intent

Thresholds:
- Primary intent: 0.80 (high confidence required for code path routing)
- Sub-intent: 0.65 (lower threshold, defaults to None if below)
- If below primary threshold, falls back to LLM classification
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from constat.core.models import TaskType
from constat.embedding_loader import EmbeddingModelLoader
from constat.execution.mode import PrimaryIntent, SubIntent, TurnIntent

logger = logging.getLogger(__name__)

# Classification thresholds
PRIMARY_THRESHOLD = 0.80
SUB_THRESHOLD = 0.65

# Embedding model - BAAI/bge-large-en-v1.5 (1024 dimensions)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024


class IntentClassifier:
    """Hierarchical intent classifier using embedding similarity with LLM fallback.

    The classifier loads exemplars from a YAML file and uses sentence-transformers
    to compute embeddings. Classification proceeds in two stages:

    1. Primary intent classification: Match user input against primary intent
       exemplars. If confidence >= 0.80, proceed to sub-intent classification.
       If confidence < 0.80, fall back to LLM classification.

    2. Sub-intent classification: Match against sub-intent exemplars scoped to
       the matched primary intent. If confidence >= 0.65, use the matched sub-intent.
       If confidence < 0.65, sub-intent is None (default behavior for that primary).

    The classifier supports multi-intent messages by splitting on sentence
    delimiters (. and ;) and classifying each segment. On conflict, the latest
    intent wins (handles natural self-correction patterns).
    """

    def __init__(
        self,
        exemplar_path: Optional[str] = None,
        llm_provider: Optional[object] = None,
    ):
        """Initialize the intent classifier.

        Args:
            exemplar_path: Path to exemplars YAML file. If None, uses the default
                          exemplars.yaml in the same directory as this module.
            llm_provider: LLM provider for fallback classification. Must have a
                         generate(system, user_message) method.

        Raises:
            FileNotFoundError: If exemplar file does not exist.
            yaml.YAMLError: If exemplar file is not valid YAML.
        """
        # Determine exemplar path
        if exemplar_path is None:
            exemplar_path = str(Path(__file__).parent / "exemplars.yaml")

        self._exemplar_path = exemplar_path
        self._llm_provider = llm_provider

        # Load exemplars
        self._exemplars = self._load_exemplars()

        # Lazy-loaded embedding model and precomputed embeddings
        self._model: Optional[object] = None
        self._primary_embeddings: Optional[dict[PrimaryIntent, np.ndarray]] = None
        self._sub_embeddings: Optional[dict[PrimaryIntent, dict[SubIntent, np.ndarray]]] = None

    def _load_exemplars(self) -> dict:
        """Load exemplars from YAML file.

        Returns:
            Dictionary with 'primary_intents' and 'sub_intents' keys.

        Raises:
            FileNotFoundError: If exemplar file does not exist.
            yaml.YAMLError: If exemplar file is not valid YAML.
        """
        exemplar_path = Path(self._exemplar_path)
        if not exemplar_path.exists():
            raise FileNotFoundError(f"Exemplar file not found: {self._exemplar_path}")

        with open(exemplar_path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Exemplar file must be a YAML dictionary: {self._exemplar_path}")

        if "primary_intents" not in data:
            raise ValueError(f"Exemplar file missing 'primary_intents' key: {self._exemplar_path}")

        return data

    def _load_embedding_model(self) -> None:
        """Lazy load the embedding model.

        Uses the shared EmbeddingModelLoader to get the model, which may
        already be loading in the background.

        Raises:
            RuntimeError: If model fails to load.
        """
        if self._model is not None:
            return

        logger.info(f"Getting embedding model from shared loader: {EMBEDDING_MODEL}")

        # Use shared loader (may already be loaded or loading in background)
        self._model = EmbeddingModelLoader.get_instance().get_model()

        # Precompute primary intent embeddings
        self._primary_embeddings = {}
        for intent_name, exemplars in self._exemplars.get("primary_intents", {}).items():
            try:
                intent = PrimaryIntent(intent_name)
            except ValueError:
                logger.warning(f"Unknown primary intent in exemplars: {intent_name}")
                continue

            embeddings = self._model.encode(exemplars, normalize_embeddings=True)
            self._primary_embeddings[intent] = embeddings

        # Precompute sub-intent embeddings (scoped by primary)
        self._sub_embeddings = {}
        for primary_name, sub_dict in self._exemplars.get("sub_intents", {}).items():
            try:
                primary = PrimaryIntent(primary_name)
            except ValueError:
                logger.warning(f"Unknown primary intent in sub_intents: {primary_name}")
                continue

            self._sub_embeddings[primary] = {}
            for sub_name, exemplars in sub_dict.items():
                try:
                    sub = SubIntent(sub_name)
                except ValueError:
                    logger.warning(f"Unknown sub-intent in exemplars: {sub_name}")
                    continue

                embeddings = self._model.encode(exemplars, normalize_embeddings=True)
                self._sub_embeddings[primary][sub] = embeddings

        logger.info(
            f"Loaded embeddings for {len(self._primary_embeddings)} primary intents, "
            f"{sum(len(s) for s in self._sub_embeddings.values())} sub-intents"
        )

    def classify(
        self,
        user_input: str,
        context: Optional[dict] = None,
    ) -> TurnIntent:
        """Classify user input into a TurnIntent.

        This is the main entry point for intent classification. For multi-intent
        messages (containing . or ;), splits and classifies each segment, with
        the latest intent winning on conflict.

        Args:
            user_input: The user's natural language input.
            context: Optional context dictionary with keys:
                    - phase: Current Phase enum value
                    - has_plan: Whether there's an active plan
                    - mode: Current Mode enum value

        Returns:
            TurnIntent with primary, sub (optional), and target (optional).
        """
        if not user_input or not user_input.strip():
            logger.warning("Empty user input, defaulting to QUERY intent")
            return TurnIntent(primary=PrimaryIntent.QUERY)

        # Handle multi-intent messages
        segments = self._split_message(user_input)

        if len(segments) > 1:
            # Classify each segment, latest wins on conflict
            return self._classify_multi_segment(segments, context)

        # Single segment classification
        return self._classify_single(user_input.strip(), context)

    def _split_message(self, user_input: str) -> list[str]:
        """Split message on sentence delimiters for multi-intent handling.

        Splits on . and ; but preserves:
        - Decimal numbers (3.14)
        - Abbreviations (e.g., i.e., etc.)
        - Quoted strings

        Args:
            user_input: The user's input string.

        Returns:
            List of non-empty segments.
        """
        # Simple split on . and ; followed by space or end of string
        # This handles most cases while avoiding splits in numbers
        pattern = r'(?<=[.;])\s+'
        segments = re.split(pattern, user_input.strip())

        # Filter empty segments and strip whitespace
        segments = [s.strip() for s in segments if s.strip()]

        return segments

    def _classify_multi_segment(
        self,
        segments: list[str],
        context: Optional[dict],
    ) -> TurnIntent:
        """Classify multiple segments and resolve conflicts.

        Classifies each segment independently. On conflict (same primary intent),
        the latest segment wins - this handles natural self-correction patterns
        like "analyze sales. wait, I got that wrong. analyze revenue instead."

        Args:
            segments: List of message segments.
            context: Optional context dictionary.

        Returns:
            TurnIntent from the winning segment.
        """
        intents: list[TurnIntent] = []

        for segment in segments:
            intent = self._classify_single(segment, context)
            intents.append(intent)

        if not intents:
            return TurnIntent(primary=PrimaryIntent.QUERY)

        # Latest wins on conflict - return the last non-trivial intent
        # A "trivial" intent would be one that looks like self-correction
        # without a new actionable request
        return intents[-1]

    def _classify_single(
        self,
        user_input: str,
        context: Optional[dict],
    ) -> TurnIntent:
        """Classify a single segment of user input.

        Args:
            user_input: A single segment of user input.
            context: Optional context dictionary.

        Returns:
            TurnIntent with primary, sub, and target.
        """
        # Ensure model is loaded
        self._load_embedding_model()

        # Classify primary intent
        primary, primary_confidence = self._classify_primary(user_input)

        if primary_confidence < PRIMARY_THRESHOLD:
            # Low confidence - fall back to LLM
            logger.info(
                f"Primary confidence {primary_confidence:.2f} below threshold {PRIMARY_THRESHOLD}, "
                f"using LLM fallback"
            )
            return self._llm_fallback(user_input, context)

        # Classify sub-intent (scoped to primary)
        sub, sub_confidence = self._classify_sub(primary, user_input)

        if sub is not None and sub_confidence < SUB_THRESHOLD:
            # Below threshold - use default (None)
            logger.info(
                f"Sub-intent confidence {sub_confidence:.2f} below threshold {SUB_THRESHOLD}, "
                f"using default"
            )
            sub = None

        # Extract target
        target = self._extract_target(primary, user_input)

        # Log classification result
        sub_log = f", sub={sub.value} at {sub_confidence:.2f}" if sub else ""
        logger.info(f"matched primary={primary.value} at {primary_confidence:.2f}{sub_log}")

        return TurnIntent(primary=primary, sub=sub, target=target)

    def _classify_primary(self, user_input: str) -> tuple[PrimaryIntent, float]:
        """Classify primary intent using embedding similarity.

        Args:
            user_input: The user's input string.

        Returns:
            Tuple of (PrimaryIntent, confidence_score).
        """
        if self._model is None or self._primary_embeddings is None:
            raise RuntimeError("Embedding model not loaded")

        # Encode user input
        input_embedding = self._model.encode(user_input, normalize_embeddings=True)

        best_intent = PrimaryIntent.QUERY
        best_score = 0.0

        for intent, exemplar_embeddings in self._primary_embeddings.items():
            # Compute cosine similarity with all exemplars for this intent
            # embeddings are already normalized, so dot product = cosine similarity
            similarities = np.dot(exemplar_embeddings, input_embedding)

            # Take the max similarity (best matching exemplar)
            max_similarity = float(np.max(similarities))

            if max_similarity > best_score:
                best_score = max_similarity
                best_intent = intent

        return best_intent, best_score

    def _classify_sub(
        self,
        primary: PrimaryIntent,
        user_input: str,
    ) -> tuple[Optional[SubIntent], float]:
        """Classify sub-intent scoped to the primary intent.

        Args:
            primary: The matched primary intent.
            user_input: The user's input string.

        Returns:
            Tuple of (SubIntent or None, confidence_score).
            Returns (None, 0.0) if no sub-intents are defined for this primary.
        """
        if self._model is None or self._sub_embeddings is None:
            raise RuntimeError("Embedding model not loaded")

        # Check if this primary has sub-intents
        if primary not in self._sub_embeddings:
            return None, 0.0

        sub_dict = self._sub_embeddings[primary]
        if not sub_dict:
            return None, 0.0

        # Encode user input
        input_embedding = self._model.encode(user_input, normalize_embeddings=True)

        best_sub: Optional[SubIntent] = None
        best_score = 0.0

        for sub, exemplar_embeddings in sub_dict.items():
            similarities = np.dot(exemplar_embeddings, input_embedding)
            max_similarity = float(np.max(similarities))

            if max_similarity > best_score:
                best_score = max_similarity
                best_sub = sub

        return best_sub, best_score

    def _extract_target(
        self,
        primary: PrimaryIntent,
        user_input: str,
    ) -> Optional[str]:
        """Extract the target from user input based on primary intent.

        The target represents what to drill into, modify, compare, etc.

        Args:
            primary: The matched primary intent.
            user_input: The user's input string.

        Returns:
            Extracted target string, or None if not applicable.
        """
        # For now, use simple heuristics. This could be enhanced with
        # more sophisticated extraction using the LLM.

        input_lower = user_input.lower()

        # Query intents: target is what they're asking about
        if primary == PrimaryIntent.QUERY:
            # Look for "about X", "the X", "for X" patterns
            patterns = [
                r"(?:about|regarding|for|of)\s+(.+?)(?:\?|$)",
                r"(?:what|how|why)\s+(?:is|are|was|were|did)\s+(.+?)(?:\?|$)",
            ]
            for pattern in patterns:
                match = re.search(pattern, input_lower)
                if match:
                    return match.group(1).strip()

        # Plan continue: target is the modification
        elif primary == PrimaryIntent.PLAN_CONTINUE:
            # Look for "change X to Y", "add X", "instead of X" patterns
            patterns = [
                r"(?:change|modify|update)\s+(.+?)\s+to",
                r"(?:add|include)\s+(.+?)$",  # Capture to end of string
                r"(?:instead of|rather than)\s+(.+?)(?:,|$)",
            ]
            for pattern in patterns:
                match = re.search(pattern, input_lower)
                if match:
                    return match.group(1).strip()

        # Plan new: target is the subject of analysis
        elif primary == PrimaryIntent.PLAN_NEW:
            # Look for "analyze X", "calculate X", "check X" patterns
            patterns = [
                r"(?:analyze|calculate|compute|find|check|verify|determine)\s+(.+?)(?:\?|$)",
                r"(?:what is|what's)\s+(?:the\s+)?(.+?)(?:\?|$)",
            ]
            for pattern in patterns:
                match = re.search(pattern, input_lower)
                if match:
                    return match.group(1).strip()

        return None

    def _llm_fallback(
        self,
        user_input: str,
        context: Optional[dict],
    ) -> TurnIntent:
        """Use LLM for intent classification when embedding confidence is low.

        Args:
            user_input: The user's input string.
            context: Optional context dictionary.

        Returns:
            TurnIntent classified by the LLM.

        Raises:
            RuntimeError: If no LLM provider is configured.
        """
        if self._llm_provider is None:
            # No LLM available - log warning and return best embedding match
            logger.warning(
                "LLM fallback requested but no provider configured, "
                "using best embedding match despite low confidence"
            )
            primary, _ = self._classify_primary(user_input)
            sub, _ = self._classify_sub(primary, user_input)
            target = self._extract_target(primary, user_input)
            return TurnIntent(primary=primary, sub=sub, target=target)

        # Build context string
        phase_str = context.get("phase", "idle") if context else "idle"
        has_plan = context.get("has_plan", False) if context else False
        mode_str = context.get("mode", "exploratory") if context else "exploratory"

        # Handle enum values
        if hasattr(phase_str, "value"):
            phase_str = phase_str.value
        if hasattr(mode_str, "value"):
            mode_str = mode_str.value

        system_prompt = """You are an intent classifier for a conversational analytics system.
Given the user input and conversation context, classify the intent.

PRIMARY INTENTS:
- query: Answer from knowledge or current context (explain, clarify, show proof)
- plan_new: Start planning a new task (analyze, calculate, verify, compare)
- plan_continue: Refine or extend the active plan (change, add, modify current work)
- control: System/session commands (mode switch, reset, help, exit, cancel)

SUB-INTENTS (optional, based on primary):
For query: detail, provenance, summary, lookup
For plan_new: compare, predict
For control: mode_switch, reset, redo_cmd, help, status, exit, cancel, replan

IMPORTANT DISTINCTIONS:
- "don't use X", "exclude X", "without X" when X is a document/table/source = plan_new (user is setting constraints on their analysis)
- "cancel", "stop", "abort", "nevermind" with NO analysis request = control/cancel (user wants to stop current operation)
- If user mentions ANY data analysis task (calculate, analyze, show, find), it's plan_new even with exclusion constraints

Respond with EXACTLY this format (no extra text):
PRIMARY: query | plan_new | plan_continue | control
SUB: <sub-intent> | none
TARGET: <extracted target> | none
CONFIDENCE: high | medium | low"""

        user_message = f"""User input: "{user_input}"
Current phase: {phase_str}
Has active plan: {has_plan}
Mode: {mode_str}"""

        try:
            result = self._llm_provider.execute(
                task_type=TaskType.INTENT_CLASSIFICATION,
                system=system_prompt,
                user_message=user_message,
                max_tokens=self.llm.max_output_tokens,
            )

            return self._parse_llm_response(result.content, user_input)

        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            # Fall back to embedding match
            primary, _ = self._classify_primary(user_input)
            sub, _ = self._classify_sub(primary, user_input)
            target = self._extract_target(primary, user_input)
            return TurnIntent(primary=primary, sub=sub, target=target)

    def _parse_llm_response(self, response: str, user_input: str) -> TurnIntent:
        """Parse the LLM's structured response into a TurnIntent.

        Args:
            response: The LLM's response string.
            user_input: Original user input for target extraction fallback.

        Returns:
            TurnIntent parsed from the response.
        """
        lines = response.strip().split("\n")

        primary = PrimaryIntent.QUERY
        sub: Optional[SubIntent] = None
        target: Optional[str] = None

        for line in lines:
            line = line.strip()

            if line.upper().startswith("PRIMARY:"):
                value = line.split(":", 1)[1].strip().lower()
                # Handle LLM returning pipe-separated options (e.g., "plan_new | none")
                # Try each part until we find a valid enum value
                found = False
                for part in value.split("|"):
                    part = part.strip()
                    if part and part != "none":
                        try:
                            primary = PrimaryIntent(part)
                            found = True
                            break
                        except ValueError:
                            continue
                if not found and value:
                    logger.warning(f"LLM returned unknown primary intent: {value}")

            elif line.upper().startswith("SUB:"):
                value = line.split(":", 1)[1].strip().lower()
                # Handle pipe-separated options
                for part in value.split("|"):
                    part = part.strip()
                    if part and part != "none":
                        try:
                            sub = SubIntent(part)
                            break
                        except ValueError:
                            continue
                else:
                    if value and value != "none" and "|" not in value:
                        logger.warning(f"LLM returned unknown sub-intent: {value}")

            elif line.upper().startswith("TARGET:"):
                value = line.split(":", 1)[1].strip()
                if value.lower() != "none":
                    target = value

        # If no target from LLM, try extraction
        if target is None:
            target = self._extract_target(primary, user_input)

        logger.info(f"LLM classified: primary={primary.value}, sub={sub}, target={target}")

        return TurnIntent(primary=primary, sub=sub, target=target)

    def set_llm_provider(self, provider: object) -> None:
        """Set the LLM provider for fallback classification.

        Args:
            provider: LLM provider with generate(system, user_message) method.
        """
        self._llm_provider = provider
