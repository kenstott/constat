# Copyright (c) 2025 Kenneth Stott
# Canary: 5b55f58b-0528-4dee-a56c-0a144b0a6825
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Output formatters for golden question test results."""

from __future__ import annotations

import json

from constat.testing.models import DomainTestResult, QuestionResult


def format_text(results: list[DomainTestResult]) -> str:
    """Format test results as colored terminal output."""
    lines: list[str] = []
    for dr in results:
        total = len(dr.questions)
        lines.append(
            f"{dr.domain}: {total} questions, "
            f"{dr.passed_count} passed, {dr.failed_count} failed"
        )
        lines.append("")
        for qr in dr.questions:
            mark = "[green]\u2713[/green]" if qr.passed else "[red]\u2717[/red]"
            layer_summary = "  ".join(
                f"{lr.layer}: {lr.passed}/{lr.total}" for lr in qr.layers
            )
            lines.append(f'  {mark} "{qr.question}"')
            lines.append(f"    {layer_summary}")
            for lr in qr.layers:
                for f in lr.failures:
                    lines.append(f"    [red]FAIL[/red] {lr.layer}: {f}")
        lines.append("")
    return "\n".join(lines)


def _question_to_dict(qr: QuestionResult) -> dict:
    return {
        "question": qr.question,
        "tags": qr.tags,
        "passed": qr.passed,
        "layers": [
            {
                "layer": lr.layer,
                "passed": lr.passed,
                "total": lr.total,
                "failures": lr.failures,
            }
            for lr in qr.layers
        ],
    }


def format_json(results: list[DomainTestResult]) -> str:
    """Format test results as JSON."""
    data = [
        {
            "domain": dr.domain,
            "passed_count": dr.passed_count,
            "failed_count": dr.failed_count,
            "questions": [_question_to_dict(qr) for qr in dr.questions],
        }
        for dr in results
    ]
    return json.dumps(data, indent=2)
