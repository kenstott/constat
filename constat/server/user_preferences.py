# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""User preferences management.

Stores user preferences in .constat/{user_id}/preferences.yaml including:
- selected_domains: List of domain IDs to auto-select on session create
- Other preferences can be added here in the future
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default preferences for new users
DEFAULT_PREFERENCES = {
    "selected_domains": [],
    "brief_mode": False,
    "default_mode": "exploratory",
}


def _get_preferences_path(user_id: str) -> Path:
    """Get the path to the user's preferences file."""
    return Path(".constat") / user_id / "preferences.yaml"


def load_user_preferences(user_id: str) -> dict[str, Any]:
    """Load user preferences from disk.

    Args:
        user_id: The user's ID

    Returns:
        Dictionary of user preferences, with defaults for missing keys
    """
    prefs_path = _get_preferences_path(user_id)

    if not prefs_path.exists():
        return DEFAULT_PREFERENCES.copy()

    try:
        with open(prefs_path, "r") as f:
            prefs = yaml.safe_load(f) or {}

        # Merge with defaults for any missing keys
        result = DEFAULT_PREFERENCES.copy()
        result.update(prefs)
        return result

    except Exception as e:
        logger.warning(f"Failed to load preferences for user {user_id}: {e}")
        return DEFAULT_PREFERENCES.copy()


def save_user_preferences(user_id: str, preferences: dict[str, Any]) -> bool:
    """Save user preferences to disk.

    Args:
        user_id: The user's ID
        preferences: Dictionary of preferences to save

    Returns:
        True if saved successfully, False otherwise
    """
    prefs_path = _get_preferences_path(user_id)

    try:
        # Ensure directory exists
        prefs_path.parent.mkdir(parents=True, exist_ok=True)

        with open(prefs_path, "w") as f:
            yaml.safe_dump(preferences, f, default_flow_style=False)

        return True

    except Exception as e:
        logger.error(f"Failed to save preferences for user {user_id}: {e}")
        return False


def get_selected_domains(user_id: str) -> list[str]:
    """Get the user's selected domains.

    Args:
        user_id: The user's ID

    Returns:
        List of domain IDs that should be pre-selected
    """
    prefs = load_user_preferences(user_id)
    return prefs.get("selected_domains", [])


# Backwards compatibility alias
get_selected_projects = get_selected_domains


def set_selected_domains(user_id: str, domain_ids: list[str]) -> bool:
    """Set the user's selected domains.

    Args:
        user_id: The user's ID
        domain_ids: List of domain IDs to save as selected

    Returns:
        True if saved successfully, False otherwise
    """
    prefs = load_user_preferences(user_id)
    prefs["selected_domains"] = domain_ids
    return save_user_preferences(user_id, prefs)


# Backwards compatibility alias
set_selected_projects = set_selected_domains


def update_preference(user_id: str, key: str, value: Any) -> bool:
    """Update a single preference value.

    Args:
        user_id: The user's ID
        key: Preference key to update
        value: New value

    Returns:
        True if saved successfully, False otherwise
    """
    prefs = load_user_preferences(user_id)
    prefs[key] = value
    return save_user_preferences(user_id, prefs)


def get_preference(user_id: str, key: str, default: Any = None) -> Any:
    """Get a single preference value.

    Args:
        user_id: The user's ID
        key: Preference key to get
        default: Default value if key not found

    Returns:
        The preference value, or default if not found
    """
    prefs = load_user_preferences(user_id)
    return prefs.get(key, default)
