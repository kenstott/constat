# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""AgentSelectorScreen modal dialog for selecting an agent."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static, OptionList
from textual.widgets.option_list import Option


class AgentSelectorScreen(ModalScreen[str | None]):
    """Modal screen for selecting an agent."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    AgentSelectorScreen {
        align: center middle;
    }

    AgentSelectorScreen > Vertical {
        width: 50;
        height: auto;
        max-height: 20;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    AgentSelectorScreen > Vertical > Static {
        text-align: center;
        margin-bottom: 1;
    }

    AgentSelectorScreen OptionList {
        height: auto;
        max-height: 12;
    }
    """

    def __init__(self, agents: list[str], current_agent: str | None = None):
        super().__init__()
        self.agents = agents
        self.current_agent = current_agent

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Select Agent", classes="title")
            option_list = OptionList(id="agent-list")
            option_list.add_option(Option("(no agent)", id="__none__"))
            for agent in self.agents:
                marker = "→ " if agent == self.current_agent else "  "
                option_list.add_option(Option(f"{marker}{agent}", id=agent))
            yield option_list

    def on_mount(self) -> None:
        """Focus the option list."""
        self.query_one("#agent-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle agent selection."""
        selected_id = str(event.option.id) if event.option.id else None
        if selected_id == "__none__":
            self.dismiss(None)
        else:
            self.dismiss(selected_id)

    def action_cancel(self) -> None:
        """Cancel without changing agent."""
        self.dismiss(self.current_agent)
