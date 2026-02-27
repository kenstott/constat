# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Interactive REPL for refinement loop."""

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
    "/export", "/download-code",
    "/database", "/databases", "/db", "/file", "/files",
    "/doc", "/docs", "/documents", "/api", "/apis",
    "/discover", "/correct", "/learnings", "/compact-learnings", "/forget-learning",
    "/rule", "/rule-edit", "/rule-delete",
    "/audit", "/summarize", "/prove",
    "/glossary", "/define", "/undefine", "/refine",
    "/schema", "/search-tables", "/search-apis", "/search-docs", "/search-chunks",
    "/lookup", "/entity", "/known-facts", "/sources",
    "/agent", "/agents", "/agent-create", "/agent-edit", "/agent-delete", "/agent-draft",
    "/skill", "/skills", "/skill-create", "/skill-edit", "/skill-delete",
    "/skill-deactivate", "/skill-draft", "/skill-download",
    "/quit", "/exit", "/q"
]


class InteractiveREPL(
    _CoreMixin,
    _DataCommandsMixin,
    _SessionCommandsMixin,
    _FactCommandsMixin,
):
    """Interactive Read-Eval-Print Loop for Constat sessions."""


def run_repl(config_path: str, verbose: bool = False, problem: str | None = None) -> None:
    """Run the interactive REPL."""
    config = Config.from_yaml(config_path)
    repl = InteractiveREPL(config, verbose=verbose)
    repl.run(initial_problem=problem)
