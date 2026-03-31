# Copyright (c) 2025 Kenneth Stott
# Canary: 34a3ce82-bb8c-4bb4-952f-c436c0b29347
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Assertion dataclasses, result types, and YAML parser for golden question testing."""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Assertion types
# ---------------------------------------------------------------------------

@dataclass
class GroundingAssertion:
    entity: str
    resolves_to: list[str]
    strict: bool = True  # False = just verify entity has any grounding


@dataclass
class RelationshipAssertion:
    subject: str
    verb: str
    object: str
    min_confidence: float = 0.5


@dataclass
class TermAssertion:
    """Unified term/entity/alias check — replaces separate entities + glossary layers."""
    name: str
    has_definition: bool = False
    domain: str | None = None
    parent: str | None = None


_DEFAULT_JUDGE_PROMPT = (
    "You are a regression test evaluator. You are given:\n"
    "1. A question that was asked\n"
    "2. The proof system's prose summary of the answer\n"
    "3. The actual computed artifacts (tables with data)\n\n"
    "The artifacts are the PRIMARY evidence. The prose summary may omit details "
    "that exist in the artifact data. If the artifacts contain the expected data "
    "(correct columns, reasonable values, appropriate row counts), the test PASSES "
    "even if the prose summary is incomplete.\n\n"
    "PASS/FAIL should be based on DATA CORRECTNESS only:\n"
    "- Expected tables exist with correct columns\n"
    "- Values are in reasonable ranges\n"
    "- Row counts match expectations\n"
    "- Computed columns are mathematically consistent\n\n"
    "Do NOT fail for:\n"
    "- Extra columns beyond those required — additional data is fine as long as "
    "the expected columns are present and correct\n"
    "- Extra tables beyond those required — auxiliary artifacts do not invalidate results\n"
    "- Duplicate values in non-key columns — rows are distinct if they have different "
    "primary keys or row identifiers. Two rows with the same product name but different "
    "IDs are distinct records, not duplicates\n"
    "- Process transparency (whether methodology is visible in output)\n"
    "- Minor floating-point display differences\n"
    "- Integer values where integers are expected (e.g., ratings 1-5)\n"
    "- Trailing zero display: 'rounded to N decimal places' means the value has "
    "AT MOST N decimal places of precision, NOT that it must display exactly N digits. "
    "0.03 and 0.030 are identical values; both satisfy 'rounded to 3 decimal places'.\n\n"
    "Reply with exactly YES or NO on the first line, then one sentence explaining why."
)


@dataclass
class EndToEndAssertion:
    """Phase 2 — run question through real pipeline, LLM-judge evaluates."""
    result_contains: list[str] = field(default_factory=list)
    judge_prompt: str = _DEFAULT_JUDGE_PROMPT
    validator_code: str = ""  # Python code that validates tables; raise AssertionError to fail
    plan_min_steps: int = 1
    expect_success: bool = True


@dataclass
class ExpectedOutput:
    """Expected output artifact — table, image, document, or markdown."""
    name: str
    type: str = "table"  # table, image, document, markdown, json, xml, pdf, ...
    columns: list[str] = field(default_factory=list)  # only for type=table


@dataclass
class GoldenExpectations:
    terms: list[TermAssertion] = field(default_factory=list)
    grounding: list[GroundingAssertion] = field(default_factory=list)
    relationships: list[RelationshipAssertion] = field(default_factory=list)
    expected_outputs: list[ExpectedOutput] = field(default_factory=list)
    end_to_end: EndToEndAssertion | None = None


@dataclass
class GoldenQuestion:
    question: str
    tags: list[str]
    expect: GoldenExpectations
    objectives: list[str] = field(default_factory=list)  # original question + follow-ups
    step_hints: list[dict] = field(default_factory=list)  # reference code from exploratory session
    system_prompt: str = ""  # domain context captured at test creation time


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EndToEndResult:
    passed: bool
    answer: str | None = None
    judge_reasoning: str | None = None
    failures: list[str] = field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class LayerResult:
    layer: str
    passed: int
    total: int
    failures: list[str] = field(default_factory=list)


@dataclass
class QuestionResult:
    question: str
    tags: list[str]
    layers: list[LayerResult] = field(default_factory=list)
    end_to_end: EndToEndResult | None = None

    @property
    def passed(self) -> bool:
        layers_pass = all(lr.passed == lr.total for lr in self.layers)
        e2e_pass = self.end_to_end.passed if self.end_to_end else True
        return layers_pass and e2e_pass


@dataclass
class DomainTestResult:
    domain: str
    questions: list[QuestionResult] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for q in self.questions if q.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for q in self.questions if not q.passed)


# ---------------------------------------------------------------------------
# YAML parser
# ---------------------------------------------------------------------------

def _parse_expectations(raw: dict) -> GoldenExpectations:
    # Merge legacy "entities" and "glossary" into unified "terms"
    terms: list[TermAssertion] = []
    seen_terms: set[str] = set()
    for t in raw.get("terms", []):
        name = t["name"]
        if name not in seen_terms:
            seen_terms.add(name)
            terms.append(TermAssertion(
                name=name,
                has_definition=t.get("has_definition", False),
                domain=t.get("domain"),
                parent=t.get("parent"),
            ))
    # Legacy: convert old "entities" list into TermAssertions
    for e in raw.get("entities", []):
        if e not in seen_terms:
            seen_terms.add(e)
            terms.append(TermAssertion(name=e))
    # Legacy: convert old "glossary" list into TermAssertions
    for gl in raw.get("glossary", []):
        name = gl["name"]
        if name not in seen_terms:
            seen_terms.add(name)
            terms.append(TermAssertion(
                name=name,
                has_definition=gl.get("has_definition", False),
                domain=gl.get("domain"),
                parent=gl.get("parent"),
            ))

    grounding = [
        GroundingAssertion(entity=g["entity"], resolves_to=g.get("resolves_to", []), strict=g.get("strict", True))
        for g in raw.get("grounding", [])
    ]
    relationships = [
        RelationshipAssertion(
            subject=r["subject"],
            verb=r["verb"],
            object=r["object"],
            min_confidence=r.get("min_confidence", 0.5),
        )
        for r in raw.get("relationships", [])
    ]
    expected_outputs = [
        ExpectedOutput(name=o["name"], type=o.get("type", "table"), columns=o.get("columns", []))
        for o in raw.get("expected_outputs", [])
    ]
    e2e_raw = raw.get("end_to_end")
    end_to_end = None
    if e2e_raw:
        judge_prompt = e2e_raw.get("judge_prompt", "")
        # Legacy: if old semantic_match exists but no judge_prompt, fold it in
        legacy_criteria = e2e_raw.get("semantic_match")
        if legacy_criteria and not judge_prompt:
            judge_prompt = f"{_DEFAULT_JUDGE_PROMPT}\n\nCriteria: {legacy_criteria}"
        end_to_end = EndToEndAssertion(
            result_contains=e2e_raw.get("result_contains", []),
            judge_prompt=judge_prompt or _DEFAULT_JUDGE_PROMPT,
            validator_code=e2e_raw.get("validator_code", ""),
            plan_min_steps=e2e_raw.get("plan_min_steps", 1),
            expect_success=e2e_raw.get("expect_success", True),
        )
    return GoldenExpectations(
        terms=terms,
        grounding=grounding,
        relationships=relationships,
        expected_outputs=expected_outputs,
        end_to_end=end_to_end,
    )


def parse_golden_questions(raw: list[dict]) -> list[GoldenQuestion]:
    """Parse raw YAML dicts into typed GoldenQuestion objects."""
    results = []
    for item in raw:
        expect_raw = item.get("expect", {})
        results.append(
            GoldenQuestion(
                question=item["question"],
                tags=item.get("tags", []),
                expect=_parse_expectations(expect_raw),
                objectives=item.get("objectives", []),
                step_hints=item.get("step_hints", []),
                system_prompt=item.get("system_prompt", ""),
            )
        )
    return results
