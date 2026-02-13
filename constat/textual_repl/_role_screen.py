# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""RoleSelectorScreen modal dialog for selecting a role."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static, OptionList
from textual.widgets.option_list import Option
from textual.screen import ModalScreen


class RoleSelectorScreen(ModalScreen[str | None]):
    """Modal screen for selecting a role."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    RoleSelectorScreen {
        align: center middle;
    }

    RoleSelectorScreen > Vertical {
        width: 50;
        height: auto;
        max-height: 20;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    RoleSelectorScreen > Vertical > Static {
        text-align: center;
        margin-bottom: 1;
    }

    RoleSelectorScreen OptionList {
        height: auto;
        max-height: 12;
    }
    """

    def __init__(self, roles: list[str], current_role: str | None = None):
        super().__init__()
        self.roles = roles
        self.current_role = current_role

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Select Role", classes="title")
            option_list = OptionList(id="role-list")
            option_list.add_option(Option("(no role)", id="__none__"))
            for role in self.roles:
                marker = "→ " if role == self.current_role else "  "
                option_list.add_option(Option(f"{marker}{role}", id=role))
            yield option_list

    def on_mount(self) -> None:
        """Focus the option list."""
        self.query_one("#role-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle role selection."""
        selected_id = str(event.option.id) if event.option.id else None
        if selected_id == "__none__":
            self.dismiss(None)
        else:
            self.dismiss(selected_id)

    def action_cancel(self) -> None:
        """Cancel without changing role."""
        self.dismiss(self.current_role)
