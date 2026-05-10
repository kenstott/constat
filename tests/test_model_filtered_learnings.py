# Copyright (c) 2025 Kenneth Stott
# Canary: 40d3bad3-8c57-41a1-b55e-001333f7df38
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for model-family-filtered codegen learnings."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from constat.storage.learnings import LearningCategory, LearningSource, LearningStore


@pytest.fixture
def store(tmp_path):
    return LearningStore(base_dir=tmp_path, user_id="test")


class TestModelFamilyFilter:

    def test_no_filter_returns_all(self, store):
        store.save_learning(
            LearningCategory.CODEGEN_ERROR,
            {"error_provider": "anthropic", "error_model": "claude-sonnet"},
            "Fix A",
        )
        store.save_learning(
            LearningCategory.CODEGEN_ERROR,
            {"error_provider": "ollama", "error_model": "llama3"},
            "Fix B",
        )
        results = store.list_raw_learnings(category=LearningCategory.CODEGEN_ERROR)
        assert len(results) == 2

    def test_filter_by_family(self, store):
        store.save_learning(
            LearningCategory.CODEGEN_ERROR,
            {"error_provider": "anthropic", "error_model": "claude-sonnet"},
            "anthropic fix",
        )
        store.save_learning(
            LearningCategory.CODEGEN_ERROR,
            {"error_provider": "ollama", "error_model": "llama3"},
            "ollama fix",
        )
        results = store.list_raw_learnings(
            category=LearningCategory.CODEGEN_ERROR,
            model_family="ollama",
        )
        assert len(results) == 1
        assert results[0]["correction"] == "ollama fix"

    def test_learnings_without_model_info_always_included(self, store):
        # Legacy learning without model info
        store.save_learning(
            LearningCategory.CODEGEN_ERROR,
            {"error_message": "KeyError: 'col'"},
            "legacy fix",
        )
        store.save_learning(
            LearningCategory.CODEGEN_ERROR,
            {"error_provider": "anthropic", "error_model": "claude-sonnet"},
            "anthropic fix",
        )
        results = store.list_raw_learnings(
            category=LearningCategory.CODEGEN_ERROR,
            model_family="anthropic",
        )
        assert len(results) == 2
        corrections = {r["correction"] for r in results}
        assert "legacy fix" in corrections
        assert "anthropic fix" in corrections

    def test_filter_excludes_other_families(self, store):
        store.save_learning(
            LearningCategory.CODEGEN_ERROR,
            {"error_provider": "openai", "error_model": "gpt-4"},
            "openai fix",
        )
        results = store.list_raw_learnings(
            category=LearningCategory.CODEGEN_ERROR,
            model_family="anthropic",
        )
        assert len(results) == 0

    def test_fixed_by_fields_stored(self, store):
        lid = store.save_learning(
            LearningCategory.CODEGEN_ERROR,
            {
                "error_provider": "ollama",
                "error_model": "llama3",
                "fixed_by_provider": "anthropic",
                "fixed_by_model": "claude-sonnet",
            },
            "escalation fix",
        )
        learning = store.get_learning(lid)
        assert learning["context"]["fixed_by_provider"] == "anthropic"
        assert learning["context"]["fixed_by_model"] == "claude-sonnet"
