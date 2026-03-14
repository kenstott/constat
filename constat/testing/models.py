# Copyright (c) 2025 Kenneth Stott
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
class GlossaryAssertion:
    name: str
    has_definition: bool = True
    domain: str | None = None
    parent: str | None = None


@dataclass
class EndToEndAssertion:
    """Phase 2 — run question through real pipeline, LLM-judge evaluates."""
    result_contains: list[str] = field(default_factory=list)
    semantic_match: str | None = None
    plan_min_steps: int = 1
    expect_success: bool = True


@dataclass
class GoldenExpectations:
    entities: list[str] = field(default_factory=list)
    grounding: list[GroundingAssertion] = field(default_factory=list)
    relationships: list[RelationshipAssertion] = field(default_factory=list)
    glossary: list[GlossaryAssertion] = field(default_factory=list)
    end_to_end: EndToEndAssertion | None = None


@dataclass
class GoldenQuestion:
    question: str
    tags: list[str]
    expect: GoldenExpectations
    objectives: list[str] = field(default_factory=list)  # original question + follow-ups
    step_hints: list[dict] = field(default_factory=list)  # reference code from exploratory session


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
    glossary = [
        GlossaryAssertion(
            name=gl["name"],
            has_definition=gl.get("has_definition", True),
            domain=gl.get("domain"),
            parent=gl.get("parent"),
        )
        for gl in raw.get("glossary", [])
    ]
    e2e_raw = raw.get("end_to_end")
    end_to_end = None
    if e2e_raw:
        end_to_end = EndToEndAssertion(
            result_contains=e2e_raw.get("result_contains", []),
            semantic_match=e2e_raw.get("semantic_match"),
            plan_min_steps=e2e_raw.get("plan_min_steps", 1),
            expect_success=e2e_raw.get("expect_success", True),
        )
    return GoldenExpectations(
        entities=raw.get("entities", []),
        grounding=grounding,
        relationships=relationships,
        glossary=glossary,
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
            )
        )
    return results
