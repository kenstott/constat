# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Resolution mixin: tiered resolution pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ._types import (
    Fact,
    FactSource,
    Tier2AssessmentResult,
    Tier2Strategy,
    TIER2_ASSESSMENT_PROMPT,
)

if TYPE_CHECKING:
    from . import FactResolver

logger = logging.getLogger(__name__)


class ResolutionMixin:

    def resolve_tiered(
        self: "FactResolver",
        fact_name: str,
        fact_description: str = "",
        **params,
    ) -> tuple[Fact, Optional[Tier2AssessmentResult]]:
        """
        Resolve a fact using the tiered resolution architecture.

        Tier 1: Parallel local sources (cache, config, rules, docs, database)
        Tier 2: LLM assessment (DERIVABLE, KNOWN, or USER_REQUIRED)

        Args:
            fact_name: The fact to resolve
            fact_description: Human-readable description of what this fact represents
            **params: Parameters for the fact

        Returns:
            Tuple of (Fact, Tier2AssessmentResult or None)
            - If Tier 1 succeeds: (resolved_fact, None)
            - If Tier 2 needed: (fact_or_unresolved, assessment_result)
        """
        import logging
        import time
        logger = logging.getLogger(__name__)

        cache_key = self._cache_key(fact_name, params)
        logger.info(f"[TIERED] Starting tiered resolution for: {cache_key}")
        logger.debug(f"resolve_tiered called for: {fact_name}, tier1_sources: {[s.value for s in self.strategy.tier1_sources]}")

        # Emit fact_start event for DAG visualization
        self._emit_event("fact_start", {
            "fact_name": cache_key,
            "fact_description": fact_description,
            "parameters": params,
            "status": "pending",
        })

        # Quick cache check BEFORE parallel race (avoids unnecessary work)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached and cached.is_resolved and cached.confidence >= self.strategy.min_confidence:
                logger.info(f"[TIERED] Cache hit for {cache_key}")
                self.resolution_log.append(cached)  # Log cache hits for audit trail
                return cached, None
            elif cached and cached.is_resolved:
                logger.debug(f"[TIERED] Cache hit but confidence {cached.confidence} < {self.strategy.min_confidence}")
                # Fall through to try other sources

        # ═══════════════════════════════════════════════════════════════════
        # TIER 1: Parallel Local Sources
        # ═══════════════════════════════════════════════════════════════════
        self._emit_event("fact_planning", {
            "fact_name": cache_key,
            "planning_type": "tier1_parallel",
            "sources": [s.value for s in self.strategy.tier1_sources],
            "status": "planning",
        })

        tier1_start = time.time()
        tier1_result = self._resolve_tier1_parallel(fact_name, params, cache_key)
        tier1_elapsed = time.time() - tier1_start

        if tier1_result and tier1_result.is_resolved:
            logger.info(f"[TIERED] Tier 1 resolved {cache_key} in {tier1_elapsed:.2f}s: {tier1_result.value}")
            self._emit_event("fact_resolved", {
                "fact_name": cache_key,
                "value": tier1_result.value,
                "source": tier1_result.source.value if tier1_result.source else "unknown",
                "confidence": tier1_result.confidence,
                "tier": 1,
                "elapsed_ms": int(tier1_elapsed * 1000),
                "status": "resolved",
            })
            return tier1_result, None

        logger.info(f"[TIERED] Tier 1 failed for {cache_key} after {tier1_elapsed:.2f}s, proceeding to Tier 2")

        # ═══════════════════════════════════════════════════════════════════
        # TIER 2: LLM Assessment
        # ═══════════════════════════════════════════════════════════════════
        self._emit_event("fact_planning", {
            "fact_name": cache_key,
            "planning_type": "tier2_assessment",
            "reason": "tier1_failed",
            "status": "planning",
        })

        assessment = self._assess_tier2_strategy(fact_name, fact_description, params)

        if assessment is None:
            # LLM assessment failed - return unresolved
            logger.warning(f"[TIERED] Tier 2 assessment failed for {cache_key}")
            self._emit_event("fact_failed", {
                "fact_name": cache_key,
                "reason": "tier2_assessment_failed",
                "status": "failed",
            })
            unresolved = Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning="Tier 1 failed, Tier 2 assessment failed",
            )
            self.resolution_log.append(unresolved)  # Log unresolved facts too
            return unresolved, None

        logger.info(f"[TIERED] Tier 2 assessment: {assessment.strategy.value} (confidence: {assessment.confidence})")

        # Handle based on assessment strategy
        if assessment.strategy == Tier2Strategy.KNOWN:
            # LLM provided the answer directly
            fact = Fact(
                name=cache_key,
                value=assessment.value,
                confidence=assessment.confidence,
                source=FactSource.LLM_KNOWLEDGE,
                reasoning=f"LLM knowledge: {assessment.reasoning}",
                context=assessment.caveat,
            )
            self._cache[cache_key] = fact
            self.resolution_log.append(fact)
            self._emit_event("fact_resolved", {
                "fact_name": cache_key,
                "value": assessment.value,
                "source": "llm_knowledge",
                "confidence": assessment.confidence,
                "tier": 2,
                "strategy": "known",
                "status": "resolved",
            })
            return fact, assessment

        elif assessment.strategy == Tier2Strategy.DERIVABLE:
            # Attempt derivation with the formula
            self._emit_event("fact_executing", {
                "fact_name": cache_key,
                "execution_type": "derivation",
                "formula": assessment.formula,
                "status": "executing",
            })
            derived_fact = self._execute_derivation(
                fact_name, params, cache_key, assessment
            )
            if derived_fact and derived_fact.is_resolved:
                self._emit_event("fact_resolved", {
                    "fact_name": cache_key,
                    "value": derived_fact.value,
                    "source": derived_fact.source.value if derived_fact.source else "derived",
                    "confidence": derived_fact.confidence,
                    "tier": 2,
                    "strategy": "derivable",
                    "dependencies": [f.name for f in derived_fact.because] if derived_fact.because else [],
                    "status": "resolved",
                })
                return derived_fact, assessment
            # Derivation failed - return assessment for caller to handle
            self._emit_event("fact_failed", {
                "fact_name": cache_key,
                "reason": "derivation_failed",
                "formula": assessment.formula,
                "status": "failed",
            })
            unresolved = Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning=f"Derivation failed: {assessment.formula}",
            )
            return unresolved, assessment

        elif assessment.strategy == Tier2Strategy.USER_REQUIRED:
            # Return unresolved with assessment - session layer handles user prompt
            unresolved = Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning=f"User input required: {assessment.question}",
            )
            return unresolved, assessment

        # Fallback
        unresolved = Fact(
            name=cache_key,
            value=None,
            confidence=0.0,
            source=FactSource.UNRESOLVED,
            reasoning=f"Unknown Tier 2 strategy: {assessment.strategy}",
        )
        return unresolved, assessment

    def _resolve_tier1_parallel(
        self: "FactResolver",
        fact_name: str,
        params: dict,
        cache_key: str,
    ) -> Optional[Fact]:
        """
        Tier 1: Race all local sources in parallel with timeout.

        Sources: cache, config, rules, documents, database
        All run concurrently, first successful result wins.
        """
        import logging
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
        import time
        logger = logging.getLogger(__name__)

        timeout = self.strategy.tier1_timeout
        sources = self.strategy.tier1_sources
        logger.debug(f"[TIER1] Racing sources: {[s.value for s in sources]} with {timeout}s timeout")

        def try_source(source: FactSource) -> tuple[FactSource, Optional[Fact], float]:
            """Try a single source, return (source, fact, elapsed_time)."""
            start = time.time()
            try:
                fact = self._try_resolve(source, fact_name, params, cache_key)
                elapsed = time.time() - start
                if fact:
                    logger.info(f"[TIER1] {source.value} returned fact: resolved={fact.is_resolved}, value_type={type(fact.value).__name__}")
                else:
                    logger.info(f"[TIER1] {source.value} returned None")
                return source, fact, elapsed
            except Exception as e:
                elapsed = time.time() - start
                import traceback
                logger.warning(f"[TIER1] {source.value} raised {type(e).__name__}: {e}")
                logger.debug(f"[TIER1] {source.value} traceback: {traceback.format_exc()}")
                return source, None, elapsed

        results: list[tuple[FactSource, Fact, float]] = []
        sources_tried: list[str] = []

        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            futures = {executor.submit(try_source, s): s for s in sources}

            try:
                for future in as_completed(futures, timeout=timeout):
                    source, fact, elapsed = future.result()

                    if fact is None:
                        sources_tried.append(f"{source.value}:no_result({elapsed:.2f}s)")
                        logger.debug(f"[TIER1] {source.value}: no result in {elapsed:.2f}s")
                    elif not fact.is_resolved:
                        sources_tried.append(f"{source.value}:unresolved({elapsed:.2f}s)")
                        logger.debug(f"[TIER1] {source.value}: unresolved in {elapsed:.2f}s")
                    elif fact.confidence < self.strategy.min_confidence:
                        sources_tried.append(f"{source.value}:low_conf({fact.confidence:.2f})")
                        logger.debug(f"[TIER1] {source.value}: low confidence {fact.confidence}")
                    else:
                        # Valid result
                        sources_tried.append(f"{source.value}:SUCCESS({elapsed:.2f}s)")
                        results.append((source, fact, elapsed))
                        logger.debug(f"[TIER1] {source.value}: success in {elapsed:.2f}s, conf={fact.confidence}")

            except TimeoutError:
                logger.warning(f"[TIER1] Timeout after {timeout}s, using available results")
                # Cancel remaining futures
                for future in futures:
                    if not future.done():
                        future.cancel()

        if not results:
            logger.debug(f"[TIER1] No valid results. Tried: {' → '.join(sources_tried)}")
            return None

        # Pick best result: highest confidence, then by source priority
        source_priority = {s: i for i, s in enumerate(sources)}
        results.sort(key=lambda x: (-x[1].confidence, source_priority.get(x[0], 999)))

        best_source, best_fact, best_elapsed = results[0]
        logger.info(f"[TIER1] Selected {best_source.value} (conf={best_fact.confidence:.2f}, {best_elapsed:.2f}s)")

        # Cache and log
        self._cache[cache_key] = best_fact
        self.resolution_log.append(best_fact)
        return best_fact

    def _assess_tier2_strategy(
        self: "FactResolver",
        fact_name: str,
        fact_description: str,
        _params: dict,
    ) -> Optional[Tier2AssessmentResult]:
        """
        Tier 2: LLM assessment of best resolution strategy.

        Returns DERIVABLE (with formula), KNOWN (with value), or USER_REQUIRED.
        """
        import logging
        import json
        logger = logging.getLogger(__name__)

        if not self.llm:
            logger.warning("[TIER2] No LLM configured, cannot assess")
            return None

        # Build context from resolution context (set by session)
        ctx = getattr(self, "_resolution_context", {})
        resolved_premises = ctx.get("resolved_premises", {})
        pending_premises = ctx.get("pending_premises", [])
        available_sources = ctx.get("available_sources", "")

        # Format resolved premises
        resolved_str = "\n".join([
            f"  - {pid}: {fact.name} = {str(fact.value)[:100]} (source: {fact.source.value})"
            for pid, fact in resolved_premises.items()
        ]) or "  (none yet)"

        # Format pending premises
        pending_str = "\n".join([
            f"  - {p.get('id', '?')}: {p.get('name', '?')} ({p.get('description', '')})"
            for p in pending_premises
        ]) or "  (none)"

        # Build prompt
        prompt = TIER2_ASSESSMENT_PROMPT.format(
            fact_name=fact_name,
            fact_description=fact_description or fact_name,
            resolved_premises=resolved_str,
            pending_premises=pending_str,
            available_sources=available_sources or "(see system context)",
        )

        logger.debug(f"[TIER2] Assessment prompt:\n{prompt}")

        response = ""
        try:
            response = self.llm.generate(
                system="You assess fact resolution strategies. Respond only with valid JSON.",
                user_message=prompt,
                max_tokens=self.llm.max_output_tokens,
            )

            # Parse JSON response
            # Handle markdown code blocks if present
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            data = json.loads(response_text)

            strategy_str = data.get("strategy", "").upper()
            if strategy_str == "DERIVABLE":
                strategy = Tier2Strategy.DERIVABLE
            elif strategy_str == "KNOWN":
                strategy = Tier2Strategy.KNOWN
            elif strategy_str == "USER_REQUIRED":
                strategy = Tier2Strategy.USER_REQUIRED
            else:
                logger.warning(f"[TIER2] Unknown strategy: {strategy_str}")
                return None

            # Validate DERIVABLE has 2+ inputs
            if strategy == Tier2Strategy.DERIVABLE:
                inputs = data.get("inputs", [])
                if self.strategy.require_multi_input_derivation and len(inputs) < 2:
                    logger.warning(f"[TIER2] DERIVABLE rejected: only {len(inputs)} inputs (need 2+)")
                    # Downgrade to USER_REQUIRED
                    return Tier2AssessmentResult(
                        strategy=Tier2Strategy.USER_REQUIRED,
                        confidence=0.5,
                        reasoning=f"Derivation rejected: needs 2+ inputs, got {len(inputs)}. User input required.",
                        question=f"What is the value for '{fact_name}'? ({fact_description})",
                    )

            return Tier2AssessmentResult(
                strategy=strategy,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                formula=data.get("formula"),
                inputs=data.get("inputs"),
                value=data.get("value"),
                caveat=data.get("caveat"),
                question=data.get("question"),
            )

        except json.JSONDecodeError as e:
            logger.error(f"[TIER2] Failed to parse JSON: {e}\nResponse: {response}")
            return None
        except Exception as e:
            logger.error(f"[TIER2] Assessment failed: {e}")
            return None

    def _execute_derivation(
        self: "FactResolver",
        _fact_name: str,
        params: dict,
        cache_key: str,
        assessment: Tier2AssessmentResult,
    ) -> Optional[Fact]:
        """
        Execute a derivation based on Tier 2 assessment.

        Resolves the input facts and applies the formula.
        """
        import logging
        logger = logging.getLogger(__name__)

        if not assessment.formula or not assessment.inputs:
            logger.warning("[DERIVATION] No formula or inputs provided")
            return None

        if len(assessment.inputs) < 2:
            logger.warning("[DERIVATION] Derivation requires 2+ inputs")
            return None

        logger.info(f"[DERIVATION] Executing: {assessment.formula}")
        logger.info(f"[DERIVATION] Inputs: {assessment.inputs}")

        # Resolve each input
        resolved_inputs: dict[str, Fact] = {}
        ctx = getattr(self, "_resolution_context", {})
        resolved_premises = ctx.get("resolved_premises", {})

        for input_name, source in assessment.inputs:
            # Check if it's a reference to an existing premise
            if source.startswith("premise:"):
                premise_id = source.split(":")[1]
                if premise_id in resolved_premises:
                    resolved_inputs[input_name] = resolved_premises[premise_id]
                    continue

            # Check if it's LLM knowledge
            if source == "llm_knowledge":
                # Resolve via LLM knowledge
                knowledge_fact = self._resolve_from_llm(input_name, params)
                if knowledge_fact:
                    resolved_inputs[input_name] = knowledge_fact
                    continue

            # Try to resolve from other sources
            # Use non-tiered resolve to avoid infinite recursion
            self._resolution_depth += 1
            try:
                fact = self._resolve_legacy(input_name, params)
                if fact and fact.is_resolved:
                    resolved_inputs[input_name] = fact
            finally:
                self._resolution_depth -= 1

        # Check if all inputs resolved
        missing = [name for name, _ in assessment.inputs if name not in resolved_inputs]
        if missing:
            logger.warning(f"[DERIVATION] Failed to resolve inputs: {missing}")
            return None

        # Execute the formula
        # Build execution context with resolved values
        exec_context = {
            name: fact.value for name, fact in resolved_inputs.items()
        }

        try:
            # Simple formula evaluation
            # Security: only allow basic math operations
            allowed_names = {"__builtins__": {"min": min, "max": max, "sum": sum, "len": len, "abs": abs}}
            allowed_names.update(exec_context)

            result = eval(assessment.formula, allowed_names)

            # Calculate confidence as min of inputs
            min_confidence = min(f.confidence for f in resolved_inputs.values())

            fact = Fact(
                name=cache_key,
                value=result,
                confidence=min_confidence * 0.95,  # Slight reduction for derivation
                source=FactSource.DERIVED,
                reasoning=f"Derived: {assessment.formula}",
                because=list(resolved_inputs.values()),
            )

            self._cache[cache_key] = fact
            self.resolution_log.append(fact)
            return fact

        except Exception as e:
            logger.error(f"[DERIVATION] Formula execution failed: {e}")
            return None

    def _resolve_legacy(self: "FactResolver", fact_name: str, params: dict) -> Fact:
        """Legacy sequential resolution (used by derivation to avoid recursion)."""
        cache_key = self._cache_key(fact_name, params)

        # Try each source in order
        for source in self.strategy.source_priority:
            if source == FactSource.SUB_PLAN:
                continue  # Skip sub-plan in legacy mode to avoid recursion
            try:
                fact = self._try_resolve(source, fact_name, params, cache_key)
                if fact and fact.is_resolved:
                    return fact
            except Exception as e:
                logger.debug(f"[_resolve_legacy] Source {source.value} failed for {fact_name}: {e}")
                continue

        return Fact(
            name=cache_key,
            value=None,
            confidence=0.0,
            source=FactSource.UNRESOLVED,
            reasoning="Legacy resolution failed",
        )

    def resolve(self: "FactResolver", fact_name: str, **params) -> Fact:
        """
        Resolve a fact by name, trying sources in priority order.

        Args:
            fact_name: The fact to resolve (e.g., "customer_ltv", "revenue_threshold")
            **params: Parameters for the fact (e.g., customer_id="acme")

        Returns:
            Fact with value, confidence, and provenance
        """
        import logging
        logger = logging.getLogger(__name__)

        # Use tiered resolution if enabled
        if self.strategy.use_tiered_resolution:
            fact, assessment = self.resolve_tiered(fact_name, **params)
            # Note: assessment contains Tier 2 result if needed by caller
            # For backward compatibility, just return the fact
            return fact

        # Legacy sequential resolution below
        # Build cache key from name + params
        cache_key = self._cache_key(fact_name, params)
        logger.debug(f"[FACT_RESOLVER] Resolving: {cache_key}")

        # Separate sources by cost/speed:
        # - Fast: CACHE, RULE, CONFIG (sync, instant)
        # - Cheap I/O: DATABASE, DOCUMENT (can parallelize)
        # - Expensive: LLM_KNOWLEDGE (API cost + latency, use as fallback)
        fast_sources = {FactSource.CACHE, FactSource.RULE, FactSource.CONFIG}
        cheap_io_sources = {FactSource.DATABASE, FactSource.DOCUMENT, FactSource.SUB_PLAN}
        expensive_sources = {FactSource.LLM_KNOWLEDGE}

        sources_tried = []

        # Phase 1: Try fast sources first (serial, quick)
        for source in self.strategy.source_priority:
            if source not in fast_sources:
                continue
            logger.debug(f"[FACT_RESOLVER] Trying fast source: {source.value}")
            try:
                fact = self._try_resolve(source, fact_name, params, cache_key)
                if fact and fact.is_resolved and fact.confidence >= self.strategy.min_confidence:
                    sources_tried.append(f"{source.value}:SUCCESS")
                    self._cache[cache_key] = fact
                    self.resolution_log.append(fact)
                    logger.info(f"[FACT_RESOLVER] Resolved {cache_key} via {source.value}: {fact.value}")
                    return fact
                elif fact is None:
                    sources_tried.append(f"{source.value}:no_result")
                elif not fact.is_resolved:
                    sources_tried.append(f"{source.value}:unresolved")
                else:
                    sources_tried.append(f"{source.value}:low_conf({fact.confidence})")
            except Exception as e:
                sources_tried.append(f"{source.value}:ERROR({type(e).__name__})")
                logger.debug(f"[FACT_RESOLVER] Error in {source.value}: {e}")

        # Phase 2: Try cheap I/O sources (DATABASE, DOCUMENT)
        cheap_io_list = [s for s in self.strategy.source_priority if s in cheap_io_sources]

        if self.strategy.parallel_io_sources and len(cheap_io_list) > 1:
            # Parallel resolution of cheap I/O sources
            logger.debug(f"[FACT_RESOLVER] Trying cheap I/O in parallel: {[s.value for s in cheap_io_list]}")
            fact = self._resolve_io_parallel(fact_name, params, cache_key, cheap_io_list, sources_tried)
            if fact:
                return fact
        else:
            # Serial cheap I/O resolution
            for source in cheap_io_list:
                logger.debug(f"[FACT_RESOLVER] Trying source: {source.value}")
                try:
                    fact = self._try_resolve(source, fact_name, params, cache_key)
                    if fact is None:
                        sources_tried.append(f"{source.value}:no_result")
                        logger.debug(f"[FACT_RESOLVER] Source {source.value} returned None - continuing to next")
                    elif not fact.is_resolved:
                        sources_tried.append(f"{source.value}:unresolved")
                        logger.debug(f"[FACT_RESOLVER] Source {source.value} returned unresolved fact - continuing")
                    elif fact.confidence < self.strategy.min_confidence:
                        sources_tried.append(f"{source.value}:low_conf({fact.confidence})")
                        logger.debug(f"[FACT_RESOLVER] Source {source.value} confidence {fact.confidence} "
                                    f"below threshold {self.strategy.min_confidence} - continuing")
                    else:
                        # Success!
                        sources_tried.append(f"{source.value}:SUCCESS")
                        self._cache[cache_key] = fact
                        self.resolution_log.append(fact)
                        logger.info(f"[FACT_RESOLVER] Resolved {cache_key} via {source.value}: {fact.value}")
                        return fact
                except Exception as e:
                    sources_tried.append(f"{source.value}:ERROR({type(e).__name__})")
                    logger.error(f"[FACT_RESOLVER] Error in source {source.value}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

        # Phase 3: Expensive fallback (LLM_KNOWLEDGE) - only if cheap sources failed
        expensive_list = [s for s in self.strategy.source_priority if s in expensive_sources]
        for source in expensive_list:
            logger.debug(f"[FACT_RESOLVER] Trying expensive fallback: {source.value}")
            try:
                fact = self._try_resolve(source, fact_name, params, cache_key)
                if fact is None:
                    sources_tried.append(f"{source.value}:no_result")
                elif not fact.is_resolved:
                    sources_tried.append(f"{source.value}:unresolved")
                elif fact.confidence < self.strategy.min_confidence:
                    sources_tried.append(f"{source.value}:low_conf({fact.confidence})")
                else:
                    sources_tried.append(f"{source.value}:SUCCESS")
                    self._cache[cache_key] = fact
                    self.resolution_log.append(fact)
                    logger.info(f"[FACT_RESOLVER] Resolved {cache_key} via {source.value}: {fact.value}")
                    return fact
            except Exception as e:
                sources_tried.append(f"{source.value}:ERROR({type(e).__name__})")
                logger.error(f"[FACT_RESOLVER] Error in expensive source {source.value}: {e}")

        # Could not resolve
        sources_summary = " → ".join(sources_tried)
        logger.debug(f"[FACT_RESOLVER] Could not resolve: {cache_key}")
        logger.debug(f"[FACT_RESOLVER] Sources tried: {sources_summary}")
        unresolved = Fact(
            name=cache_key,
            value=None,
            confidence=0.0,
            source=FactSource.UNRESOLVED,
            reasoning=f"Could not resolve fact: {fact_name}. Sources: {sources_summary}"
        )
        self.resolution_log.append(unresolved)
        return unresolved

    def _resolve_io_parallel(
        self: "FactResolver",
        fact_name: str,
        params: dict,
        cache_key: str,
        io_sources: list[FactSource],
        sources_tried: list[str],
    ) -> Optional[Fact]:
        """
        Run I/O-bound sources in parallel and pick the best result.

        Uses ThreadPoolExecutor for true parallelism in synchronous code.
        Selection: prioritizes by source order, uses confidence as tiebreaker.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import logging
        logger = logging.getLogger(__name__)

        def try_source(source: FactSource) -> tuple[FactSource, Optional[Fact]]:
            try:
                fact = self._try_resolve(source, fact_name, params, cache_key)
                return source, fact
            except Exception as e:
                logger.debug(f"[_resolve_io_parallel] {source.value} raised: {e}")
                return source, None

        # Run all I/O sources in parallel
        valid_results: list[tuple[int, float, Fact, FactSource]] = []

        with ThreadPoolExecutor(max_workers=len(io_sources)) as executor:
            futures = {executor.submit(try_source, s): s for s in io_sources}

            for future in as_completed(futures):
                source, fact = future.result()
                if fact is None:
                    sources_tried.append(f"{source.value}:no_result")
                elif not fact.is_resolved:
                    sources_tried.append(f"{source.value}:unresolved")
                elif fact.confidence < self.strategy.min_confidence:
                    sources_tried.append(f"{source.value}:low_conf({fact.confidence})")
                else:
                    # Valid result - store with priority index
                    priority_idx = io_sources.index(source)
                    valid_results.append((priority_idx, fact.confidence, fact, source))
                    sources_tried.append(f"{source.value}:conf={fact.confidence:.2f}")
                    logger.debug(f"[_resolve_io_parallel] {source.value}: conf={fact.confidence:.2f}")

        if not valid_results:
            return None

        # Pick best: sort by (priority_index, -confidence)
        valid_results.sort(key=lambda x: (x[0], -x[1]))
        best_priority, best_conf, best_fact, best_source = valid_results[0]

        sources_tried.append(f"{best_source.value}:SELECTED")
        logger.info(f"[_resolve_io_parallel] Selected {best_source.value} with confidence {best_conf:.2f}")

        self._cache[cache_key] = best_fact
        self.resolution_log.append(best_fact)
        return best_fact
