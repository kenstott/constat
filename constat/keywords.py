"""
Intent detection keywords with i18n support.

Keywords are loaded from keywords.yaml and cached on first access.

Pattern types in keywords.yaml:
- Simple strings: substring match (case-insensitive)
- Regex patterns (^...$): matched as regex against stripped input
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml


# Default language
DEFAULT_LANGUAGE = "en"

# Path to keywords file
KEYWORDS_FILE = Path(__file__).parent / "keywords.yaml"


@lru_cache(maxsize=1)
def _load_keywords() -> dict:
    """Load keywords from YAML file (cached)."""
    if not KEYWORDS_FILE.exists():
        return {}
    with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _is_regex_pattern(pattern: str) -> bool:
    """Check if pattern should be treated as regex (has anchors)."""
    return pattern.startswith("^") and pattern.endswith("$")


def get_brief_keywords(language: str = DEFAULT_LANGUAGE) -> list[str]:
    """Get keywords for brief output detection.

    Args:
        language: Language code (default: 'en')

    Returns:
        List of keywords/phrases that suggest user wants brief output
    """
    keywords = _load_keywords()
    lang_keywords = keywords.get(language, keywords.get(DEFAULT_LANGUAGE, {}))
    return lang_keywords.get("brief_output", [])


def get_mode_switch_patterns(language: str = DEFAULT_LANGUAGE) -> dict[str, list[str]]:
    """Get mode switch patterns.

    Args:
        language: Language code (default: 'en')

    Returns:
        Dict mapping mode name to list of patterns
    """
    keywords = _load_keywords()
    lang_keywords = keywords.get(language, keywords.get(DEFAULT_LANGUAGE, {}))
    return lang_keywords.get("mode_switch", {})


def _match_pattern(pattern: str, text: str) -> bool:
    """Match a pattern against text.

    Args:
        pattern: Either a simple substring or regex pattern (^...$)
        text: Text to match against (should be lowercase)

    Returns:
        True if pattern matches
    """
    if _is_regex_pattern(pattern):
        # Regex pattern - match against stripped text
        return bool(re.match(pattern, text.strip(), re.IGNORECASE))
    else:
        # Simple substring match
        return pattern in text


def wants_brief_output(query: str, language: str = DEFAULT_LANGUAGE) -> bool:
    """Check if the query suggests user wants brief output (skip insights).

    Args:
        query: User query text
        language: Language code for keyword matching

    Returns:
        True if query contains brief output keywords
    """
    query_lower = query.lower()
    keywords = get_brief_keywords(language)
    for keyword in keywords:
        if _match_pattern(keyword, query_lower):
            return True
    return False


def detect_mode_switch(text: str, language: str = DEFAULT_LANGUAGE) -> Optional[str]:
    """Detect if the user is requesting a mode switch.

    Args:
        text: User input text
        language: Language code for keyword matching

    Returns:
        Mode name if a switch is requested, None otherwise
    """
    text_lower = text.lower()
    patterns = get_mode_switch_patterns(language)

    for mode_name, mode_patterns in patterns.items():
        for pattern in mode_patterns:
            if _match_pattern(pattern, text_lower):
                return mode_name
    return None


def reload_keywords() -> None:
    """Force reload of keywords file (clears cache)."""
    _load_keywords.cache_clear()
