# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Forgiving LLM primitive wrappers.

Accept anything (str, list, Series, ndarray), deduplicate internally,
return the simplest possible type based on how they were called:
  - Single value in  → scalar out  (str / float / None)
  - Multiple values  → list out    (list[str] / list[float|None])

With reason=True, returns (value, reason) tuples instead of bare values.
"""

from __future__ import annotations

import logging

from pathlib import Path

from constat.llm import (
    llm_map as _raw_llm_map,
    llm_classify as _raw_llm_classify,
    llm_score as _raw_llm_score,
    llm_vision as _raw_llm_vision,
    llm_translate as _raw_llm_translate,
)

logger = logging.getLogger(__name__)


def _to_str_list(values) -> tuple[list[str], bool]:
    """Normalize any input to list[str]. Returns (str_list, was_scalar)."""
    if isinstance(values, str):
        return [values], True
    # pandas Series / numpy ndarray
    if hasattr(values, 'tolist'):
        items = values.tolist()
        return [str(v) for v in items], len(items) == 1
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values], len(values) == 1
    # Single non-string scalar
    return [str(values)], True


def llm_map(values, allowed, source_desc="values", target_desc="", allow_none=False, reason=False, **_kw):
    """Map values to an allowed set.

    With allow_none=False (default): every value gets a best-effort match.
    With allow_none=True: unclassifiable values return None.
    With reason=True: returns (value, reason) tuples instead of bare values.
    """
    str_values, scalar = _to_str_list(values)
    if not str_values:
        return []
    unique = list(dict.fromkeys(str_values))
    if allow_none:
        context = target_desc or source_desc
        raw = _raw_llm_classify(unique, allowed, context, reason=reason)
    else:
        raw = _raw_llm_map(unique, allowed, source_desc, target_desc, reason=reason)
    if reason:
        result = []
        for v in str_values:
            entry = raw.get(v, {})
            if isinstance(entry, dict):
                result.append((entry.get("value"), entry.get("reason", "")))
            else:
                result.append((entry, ""))
    else:
        result = [raw.get(v) for v in str_values]
    if scalar:
        return result[0]
    return result


def llm_classify(values, categories, context="", reason=False, **_kw):
    """Alias for llm_map(..., allow_none=True). Kept for backwards compatibility."""
    return llm_map(values, categories, source_desc=context, target_desc=context, allow_none=True, reason=reason, **_kw)


def llm_score(texts, min_val=0.0, max_val=1.0, instruction="Rate each text", explain=False, **_kw):
    """Score texts on a numeric scale.

    With explain=False (default): returns float|None (scalar) or list[float|None].
    With explain=True: returns (float|None, str) (scalar) or list[tuple[float|None, str]].
    """
    str_values, scalar = _to_str_list(texts)
    if not str_values:
        return []
    unique = list(dict.fromkeys(str_values))
    raw = _raw_llm_score(unique, min_val, max_val, instruction)
    score_map = dict(zip(unique, raw))
    # raw returns list[tuple[float|None, str]]
    if explain:
        result = [score_map[v] for v in str_values]
        if scalar:
            return result[0]
        return result
    result = [score_map[v][0] for v in str_values]
    if scalar:
        return result[0]
    return result


_MIME_FROM_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".svg": "image/svg+xml",
}


def llm_vision(image, prompt, mime_type=None, **_kw) -> str:
    """Analyze an image with LLM vision.

    Accepts: file path (str/Path), raw bytes, base64 string, or data URI string.
    Auto-detects mime type from file extension when not provided.
    """
    import base64 as b64

    image_bytes: bytes
    detected_mime: str | None = mime_type

    if isinstance(image, (str, Path)):
        s = str(image)
        # data URI: "data:image/png;base64,..."
        if s.startswith("data:"):
            header, data = s.split(",", 1)
            # header = "data:image/png;base64"
            detected_mime = detected_mime or header.split(";")[0].split(":")[1]
            image_bytes = b64.b64decode(data)
        elif Path(s).exists():
            # File path
            p = Path(s)
            if detected_mime is None:
                detected_mime = _MIME_FROM_EXT.get(p.suffix.lower())
            image_bytes = p.read_bytes()
        else:
            # Assume base64 string
            image_bytes = b64.b64decode(s)
    elif isinstance(image, bytes):
        image_bytes = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if detected_mime is None:
        detected_mime = "image/png"

    return _raw_llm_vision(image_bytes, detected_mime, prompt)


def llm_translate(texts, target_language, source_language=None, **_kw):
    """Translate texts to a target language.

    Accepts anything: str, list, Series, ndarray.
    Scalar in -> scalar out, list in -> list out.
    """
    str_values, scalar = _to_str_list(texts)
    if not str_values:
        return []
    unique = list(dict.fromkeys(str_values))
    raw = _raw_llm_translate(unique, target_language, source_language)
    translate_map = dict(zip(unique, raw))
    result = [translate_map[v] for v in str_values]
    if scalar:
        return result[0]
    return result
