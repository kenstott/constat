# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Interactive REPL for refinement loop."""

from typing import Optional

from constat.core.config import Config
from constat.repl.interactive._core import _CoreMixin
from constat.repl.interactive._data_commands import _DataCommandsMixin
from constat.repl.interactive._fact_commands import _FactCommandsMixin
from constat.repl.interactive._session_commands import _SessionCommandsMixin

# Commands available in the REPL
REPL_COMMANDS = [
    "/help", "/h", "/tables", "/show", "/query", "/code", "/state",
    "/update", "/refresh", "/reset", "/redo", "/user", "/save", "/share", "/sharewith",
    "/plans", "/replay", "/history", "/sessions", "/resume", "/restore",
    "/context", "/compact", "/facts", "/remember", "/forget",
    "/verbose", "/raw", "/insights", "/preferences", "/artifacts",
    "/database", "/databases", "/db", "/file", "/files",
    "/correct", "/learnings", "/compact-learnings", "/forget-learning",
    "/audit", "/summarize", "/prove",
    "/quit", "/exit", "/q"
]


class InteractiveREPL(
    _CoreMixin,
    _DataCommandsMixin,
    _SessionCommandsMixin,
    _FactCommandsMixin,
):
    """Interactive Read-Eval-Print Loop for Constat sessions."""


def run_repl(config_path: str, verbose: bool = False, problem: Optional[str] = None) -> None:
    """Run the interactive REPL."""
    config = Config.from_yaml(config_path)
    repl = InteractiveREPL(config, verbose=verbose)
    repl.run(initial_problem=problem)
