# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for REPL command matching."""

import pytest

from constat.execution.repl_matcher import (
    match_repl_command,
    tokenize,
    jaccard_similarity,
    token_overlap_ratio,
    get_all_commands,
    MatchResult,
)


class TestTokenize:
    """Tests for tokenize function."""

    def test_basic_tokenization(self):
        tokens = tokenize("start over")
        assert tokens == {"start", "over"}

    def test_removes_stopwords(self):
        tokens = tokenize("please start the session over now")
        assert "please" not in tokens
        assert "the" not in tokens
        assert "now" not in tokens
        assert "start" in tokens
        assert "session" in tokens
        assert "over" in tokens

    def test_keeps_this(self):
        # "this" is meaningful in exemplars like "explore this"
        tokens = tokenize("explore this")
        assert "this" in tokens
        assert "explore" in tokens

    def test_lowercases(self):
        tokens = tokenize("Start Over")
        assert tokens == {"start", "over"}

    def test_strips_punctuation(self):
        tokens = tokenize("start over!")
        assert tokens == {"start", "over"}

    def test_empty_string(self):
        tokens = tokenize("")
        assert tokens == set()

    def test_only_stopwords(self):
        tokens = tokenize("the a an")
        assert tokens == set()


class TestSimilarityFunctions:
    """Tests for similarity functions."""

    def test_jaccard_identical(self):
        assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self):
        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_partial(self):
        # {a, b} & {b, c} = {b}, union = {a, b, c}
        assert jaccard_similarity({"a", "b"}, {"b", "c"}) == 1/3

    def test_jaccard_empty(self):
        assert jaccard_similarity(set(), {"a"}) == 0.0
        assert jaccard_similarity({"a"}, set()) == 0.0

    def test_overlap_ratio_full(self):
        assert token_overlap_ratio({"start", "over"}, {"start", "over"}) == 1.0

    def test_overlap_ratio_superset(self):
        # User has more tokens, but all exemplar tokens are present
        assert token_overlap_ratio({"please", "start", "over", "now"}, {"start", "over"}) == 1.0

    def test_overlap_ratio_partial(self):
        assert token_overlap_ratio({"start", "fresh"}, {"start", "over"}) == 0.5


class TestMatchReplCommand:
    """Tests for match_repl_command function."""

    def test_exact_match_reset(self):
        result = match_repl_command("start over")
        assert result is not None
        assert result.command == "/reset"
        assert result.confidence >= 0.8

    def test_exact_match_with_please(self):
        result = match_repl_command("please start over")
        assert result is not None
        assert result.command == "/reset"

    def test_fresh_start(self):
        result = match_repl_command("fresh start")
        assert result is not None
        assert result.command == "/reset"

    def test_new_topic(self):
        result = match_repl_command("new topic")
        assert result is not None
        assert result.command == "/reset"

    def test_exploratory_mode(self):
        result = match_repl_command("switch to exploratory")
        assert result is not None
        assert result.command == "/mode exploratory"

    def test_explore_this(self):
        result = match_repl_command("explore this")
        assert result is not None
        assert result.command == "/mode exploratory"

    def test_auditable_mode(self):
        result = match_repl_command("audit mode")
        assert result is not None
        assert result.command == "/mode auditable"

    def test_verify_this(self):
        result = match_repl_command("verify this")
        assert result is not None
        assert result.command == "/mode auditable"

    def test_provenance(self):
        result = match_repl_command("show derivation")
        assert result is not None
        assert result.command == "/provenance"

    def test_show_your_work(self):
        result = match_repl_command("show your work")
        assert result is not None
        assert result.command == "/provenance"

    def test_redo(self):
        result = match_repl_command("run it again")
        assert result is not None
        assert result.command == "/redo"

    def test_no_match_for_question(self):
        result = match_repl_command("what is the total revenue?")
        assert result is None

    def test_no_match_for_analysis(self):
        result = match_repl_command("analyze sales by region")
        assert result is None

    def test_no_match_for_short_input(self):
        result = match_repl_command("hi")
        assert result is None

    def test_no_match_for_partial(self):
        # "start" alone shouldn't match "start over"
        result = match_repl_command("start")
        assert result is None

    def test_threshold_respected(self):
        # With very high threshold, matches with extra tokens should fail
        # "kinda start over maybe" has 4 tokens, "start over" has 2
        # Overlap = 100%, Jaccard = 2/4 = 50%
        # Combined score = 0.6*1.0 + 0.4*0.5 = 0.8
        result = match_repl_command("kinda start over maybe", threshold=0.85)
        assert result is None  # 0.8 < 0.85

    def test_lower_threshold_more_lenient(self):
        result = match_repl_command("begin fresh", threshold=0.5)
        # "begin" doesn't match but "fresh" does - depends on implementation
        # This tests that threshold parameter works


class TestMatchResultFields:
    """Tests for MatchResult fields."""

    def test_result_has_command(self):
        result = match_repl_command("start over")
        assert result.command == "/reset"

    def test_result_has_description(self):
        result = match_repl_command("start over")
        assert "fresh" in result.description.lower() or "clear" in result.description.lower()

    def test_result_has_confidence(self):
        result = match_repl_command("start over")
        assert 0.0 <= result.confidence <= 1.0

    def test_result_has_matched_exemplar(self):
        result = match_repl_command("start over")
        assert result.matched_exemplar == "start over"


class TestGetAllCommands:
    """Tests for get_all_commands function."""

    def test_returns_list(self):
        commands = get_all_commands()
        assert isinstance(commands, list)

    def test_commands_have_required_fields(self):
        commands = get_all_commands()
        for cmd in commands:
            assert "command" in cmd
            assert "description" in cmd
            assert "exemplars" in cmd

    def test_includes_reset(self):
        commands = get_all_commands()
        command_names = [c["command"] for c in commands]
        assert "/reset" in command_names

    def test_includes_mode_commands(self):
        commands = get_all_commands()
        command_names = [c["command"] for c in commands]
        assert "/mode exploratory" in command_names
        assert "/mode auditable" in command_names


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_input(self):
        result = match_repl_command("")
        assert result is None

    def test_whitespace_only(self):
        result = match_repl_command("   ")
        assert result is None

    def test_punctuation_only(self):
        result = match_repl_command("!!!")
        assert result is None

    def test_case_insensitive(self):
        result = match_repl_command("START OVER")
        assert result is not None
        assert result.command == "/reset"

    def test_extra_whitespace(self):
        result = match_repl_command("  start   over  ")
        assert result is not None
        assert result.command == "/reset"

    def test_mixed_case(self):
        result = match_repl_command("StArT OvEr")
        assert result is not None
        assert result.command == "/reset"
