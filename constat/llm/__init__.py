# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""LLM primitives for auditable data transformations.

Provides llm_enrich (unified), llm_summarize, llm_score —
importable by generated scripts. Auto-detects provider from env vars for
standalone use; call set_backend(router) for in-session use.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_backend = None  # TaskRouter | BaseLLMProvider | None
_on_call_listeners: list[Callable] = []


@dataclass
class LLMCallEvent:
    """Fired after every primitive call for tracking/auditability."""
    primitive: str
    input_count: int
    null_count: int
    model_used: str
    provider_used: str


# ---------------------------------------------------------------------------
# Backend management
# ---------------------------------------------------------------------------

def set_backend(router) -> None:
    """Set the LLM backend (TaskRouter or BaseLLMProvider) for all primitives."""
    global _backend
    _backend = router


def on_call(callback: Callable[[LLMCallEvent], None]) -> None:
    """Register a listener called after every primitive invocation."""
    _on_call_listeners.append(callback)


class _DirectProvider:
    """Lightweight LLM backend using raw provider SDKs — no constat deps."""

    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def generate(self, system: str, user_message: str, max_tokens: int = 4096) -> str:
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            # noinspection PyTypeChecker
            resp = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            return resp.content[0].text

        if self.provider == "openai":
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            # noinspection PyTypeChecker
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
            )
            return resp.choices[0].message.content

        if self.provider == "gemini":
            # noinspection PyUnresolvedReferences
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model, system_instruction=system)
            resp = model.generate_content(user_message)
            return resp.text

        # OpenAI-compatible providers (grok, together, groq, mistral)
        base_urls = {
            "grok": "https://api.x.ai/v1",
            "together": "https://api.together.xyz/v1",
            "groq": "https://api.groq.com/openai/v1",
            "mistral": "https://api.mistral.ai/v1",
        }
        if self.provider in base_urls:
            import openai
            client = openai.OpenAI(api_key=self.api_key, base_url=base_urls[self.provider])
            # noinspection PyTypeChecker
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
            )
            return resp.choices[0].message.content

        if self.provider == "ollama":
            import ollama as _ollama
            resp = _ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
            )
            return resp["message"]["content"]

        raise RuntimeError(f"Unsupported provider: {self.provider}")


def _auto_detect_backend() -> _DirectProvider:
    """Create a lightweight direct provider from the first matching env var.

    No constat.* dependencies — works standalone with just the provider SDK.
    """
    env_to_provider = [
        ("ANTHROPIC_API_KEY", "anthropic", "claude-sonnet-4-20250514"),
        ("OPENAI_API_KEY", "openai", "gpt-4o"),
        ("GOOGLE_API_KEY", "gemini", "gemini-2.0-flash"),
        ("XAI_API_KEY", "grok", "grok-3-mini-fast"),
        ("MISTRAL_API_KEY", "mistral", "mistral-large-latest"),
        ("TOGETHER_API_KEY", "together", "meta-llama/Llama-3-70b-chat-hf"),
        ("GROQ_API_KEY", "groq", "llama3-70b-8192"),
        ("OLLAMA_HOST", "ollama", "llama3"),
    ]

    for env_var, provider, model in env_to_provider:
        if os.environ.get(env_var):
            return _DirectProvider(provider, model, os.environ[env_var])

    raise RuntimeError(
        "No LLM provider found. Set one of: "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, XAI_API_KEY, "
        "MISTRAL_API_KEY, TOGETHER_API_KEY, GROQ_API_KEY, OLLAMA_HOST"
    )


def _get_backend():
    """Return the current backend, auto-detecting if needed."""
    global _backend
    if _backend is None:
        _backend = _auto_detect_backend()
    return _backend


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------

def _execute(system: str, user_message: str) -> tuple[str, str, str]:
    """Call the backend and return (content, model_used, provider_used)."""
    backend = _get_backend()

    # _DirectProvider — standalone mode, returns plain str
    if isinstance(backend, _DirectProvider):
        content = backend.generate(system=system, user_message=user_message)
        return content.strip(), backend.model, backend.provider

    # TaskRouter path (constat session) — has .execute() with task_type
    if hasattr(backend, "execute"):
        from constat.core.models import TaskType
        # noinspection PyUnresolvedReferences
        result = backend.execute(
            task_type=TaskType.SYNTHESIS,
            system=system,
            user_message=user_message,
            max_tokens=backend.max_output_tokens,
        )
        return (
            result.content.strip(),
            getattr(result, "model", "unknown"),
            getattr(result, "provider", "unknown"),
        )

    # BaseLLMProvider fallback
    # noinspection PyUnresolvedReferences
    result = backend.generate(
        system=system,
        user_message=user_message,
    )
    return (
        result.strip(),
        getattr(backend, "model", "unknown"),
        type(backend).__name__,
    )


def _parse_json(response: str) -> str:
    """Strip markdown fences from a JSON response."""
    if "```" in response:
        response = re.sub(r'```\w*\n?', '', response).strip()
    return response


def _notify(event: LLMCallEvent) -> None:
    """Fire registered listeners."""
    for cb in _on_call_listeners:
        try:
            cb(event)
        except Exception:
            logger.exception("LLM call listener error")


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def llm_map(
    values: list[str],
    allowed: list[str],
    source_desc: str = "values",
    target_desc: str = "",
    *,
    reason: bool = False,
    score: bool = False,
) -> dict[str, str] | dict[str, dict]:
    """Map a list of values to an allowed set using LLM knowledge.

    Args:
        values: List of source values to map (e.g., ["Whisker Wand", "Purrfect Pillow"])
        allowed: List of valid target values to map to (e.g., ["Abyssinian", "Bengal", ...])
        source_desc: Label for the source values (e.g., "product names")
        target_desc: Optional context about the target domain (e.g., "cat breeds")
        reason: If True, include a reasoning string per mapping.
        score: If True, include a confidence score (0.0–1.0) per mapping.

    Returns:
        dict[str, str] by default. Every input gets a best-effort mapping from allowed.
        dict[str, dict] when reason or score is True, with keys "value", "reason", "score".
        Consumer should use score to decide acceptance threshold.
    """
    import random

    values_str = "\n".join(f"- {v}" for v in values)
    allowed_str = "\n".join(f"- {a}" for a in allowed)
    allowed_set = set(allowed)
    target_ctx = f" ({target_desc})" if target_desc else ""
    rich = reason or score

    # Always ask for reason+score internally so the LLM can express
    # uncertainty via score instead of returning null.
    entry_keys = ['"value"', '"reason"', '"score"']
    keys_str = ", ".join(entry_keys)

    example_val = allowed[0] if allowed else "allowed_val"
    example_val2 = allowed[1] if len(allowed) > 1 else example_val
    example_entry = {"value": example_val, "reason": "brief explanation", "score": 0.95}
    example_low = {"value": example_val2, "reason": "weak match but closest available", "score": 0.2}
    example_json = json.dumps({"input1": example_entry, "input2": example_low})

    prompt = f"""Map each of the following {source_desc} to the most appropriate value from the ALLOWED set{target_ctx}.

SOURCE {source_desc.upper()}:
{values_str}

ALLOWED VALUES (you MUST pick from this list):
{allowed_str}

Respond with ONLY valid JSON: a single object where EVERY input is a key mapped to an object with keys {keys_str}.
CRITICAL RULES:
- "value" MUST be one of the ALLOWED VALUES above. NEVER null. If no good match exists, pick a random allowed value, set score to 0.0, and set reason to "random — no meaningful match".
- "reason" is a brief explanation of the mapping.
- "score" is a float 0.0–1.0 reflecting confidence. Low scores are expected for weak matches.
- The JSON keys MUST be the EXACT input strings listed above.

Example format: {example_json}

YOUR JSON RESPONSE:"""

    system_msg = f"You map {source_desc} to an allowed set of values{target_ctx}. Output ONLY valid JSON. Each entry has keys {keys_str}. EVERY value must come from the allowed set — NEVER null. Use score to express confidence."

    content, model_used, provider_used = _execute(
        system=system_msg,
        user_message=prompt,
    )

    content = _parse_json(content)

    if not content.startswith("{"):
        logger.warning(f"[LLM_MAP] Could not parse response: {content[:200]}")
        mapping = {v: {"value": random.choice(allowed), "reason": "parse failure fallback", "score": 0.0} for v in values}
    else:
        mapping = json.loads(content)

        # Normalize: ensure every entry is a rich dict
        for k, v in mapping.items():
            if not isinstance(v, dict):
                mapping[k] = {"value": v, "reason": "", "score": 0.5 if (v and v in allowed_set) else 0.0}

        # Identify failed entries (null or not in allowed set)
        failed = {k for k, v in mapping.items()
                  if v.get("value") is None or v.get("value") not in allowed_set}

        # Retry failed entries once
        if failed:
            logger.warning(f"[LLM_MAP] {len(failed)} null/invalid entries, retrying: {list(failed)}")
            retry_values_str = "\n".join(f"- {v}" for v in failed)
            retry_prompt = prompt.replace(f"SOURCE {source_desc.upper()}:\n{values_str}", f"SOURCE {source_desc.upper()}:\n{retry_values_str}")
            retry_content, _, _ = _execute(system=system_msg, user_message=retry_prompt)
            retry_content = _parse_json(retry_content)
            if retry_content.startswith("{"):
                retry_mapping = json.loads(retry_content)
                for k in failed:
                    if k in retry_mapping:
                        rv = retry_mapping[k]
                        if isinstance(rv, dict) and rv.get("value") in allowed_set:
                            mapping[k] = rv
                        elif isinstance(rv, str) and rv in allowed_set:
                            mapping[k] = {"value": rv, "reason": "", "score": 0.5}

        # Final safety net: any still-broken entries get a random allowed value
        for k, v in mapping.items():
            val = v.get("value") if isinstance(v, dict) else v
            if val is None or val not in allowed_set:
                if val is not None:
                    logger.warning(f"[LLM_MAP] Hallucinated value for {k!r}: {val!r}, replacing with random")
                mapping[k] = {"value": random.choice(allowed), "reason": "random — no match found after retry", "score": 0.0}

    # Strip internal reason/score if caller didn't ask for them
    if not rich:
        mapping = {k: v["value"] if isinstance(v, dict) else v for k, v in mapping.items()}
    else:
        for k, v in mapping.items():
            if isinstance(v, dict):
                if not reason:
                    v.pop("reason", None)
                if not score:
                    v.pop("score", None)

    null_count = 0  # We never return nulls now
    logger.info(
        f"[LLM_MAP] Mapped {len(values)}/{len(values)} {source_desc} -> allowed set ({len(allowed)} values)"
    )

    _notify(LLMCallEvent(
        primitive="llm_map",
        input_count=len(values),
        null_count=null_count,
        model_used=model_used,
        provider_used=provider_used,
    ))

    return mapping


def _null_rich_entry(reason: bool, score: bool) -> dict:
    """Build a null rich entry for failed parse fallback."""
    entry = {"value": None}
    if reason:
        entry["reason"] = None
    if score:
        entry["score"] = None
    return entry


def llm_classify(
    values: list[str],
    categories: list[str],
    context: str = "",
    *,
    reason: bool = False,
    score: bool = False,
) -> dict[str, str] | dict[str, dict]:
    """Classify items into fixed categories using LLM knowledge.

    Args:
        values: List of items to classify.
        categories: List of valid category names.
        context: Optional context describing the domain.
        reason: If True, include a reasoning string per classification.
        score: If True, include a confidence score (0.0–1.0) per classification.

    Returns:
        dict[str, str] by default. Unclassifiable values map to None.
        dict[str, dict] when reason or score is True, with keys "value", "reason", "score".
    """
    values_str = "\n".join(f"- {v}" for v in values)
    cats_str = ", ".join(f'"{c}"' for c in categories)
    ctx = f" ({context})" if context else ""
    rich = reason or score

    if rich:
        entry_keys = ['"value"']
        if reason:
            entry_keys.append('"reason"')
        if score:
            entry_keys.append('"score"')
        keys_str = ", ".join(entry_keys)

        example_entry = {"value": "category_a"}
        if reason:
            example_entry["reason"] = "brief explanation"
        if score:
            example_entry["score"] = 0.95
        example_null = {"value": None}
        if reason:
            example_null["reason"] = None
        if score:
            example_null["score"] = None
        example_json = json.dumps({"item1": example_entry, "item2": example_null})

        prompt = f"""Classify each of the following items{ctx} into exactly one of these categories: {cats_str}.

{values_str}

Respond with ONLY valid JSON: a single object mapping each input to an object with keys {keys_str}.
"value" must be one of the categories above, or null if unclassifiable.
{"\"reason\" should be a brief explanation of the classification." if reason else ""}
{"\"score\" should be a float 0.0–1.0 reflecting confidence in the classification." if score else ""}

Example format: {example_json}

YOUR JSON RESPONSE:"""

        system_msg = f"You classify items into categories: {cats_str}. Output ONLY valid JSON. Each entry has keys {keys_str}. Use null value for uncertain."
    else:
        prompt = f"""Classify each of the following items{ctx} into exactly one of these categories: {cats_str}.

{values_str}

Respond with ONLY valid JSON: a single object mapping each input to its category.
If an item cannot be confidently classified, map it to null.

Example format: {{"item1": "category_a", "item2": "category_b", "item3": null}}

YOUR JSON RESPONSE:"""

        system_msg = f"You classify items into categories: {cats_str}. Output ONLY valid JSON. Use null for uncertain."

    content, model_used, provider_used = _execute(
        system=system_msg,
        user_message=prompt,
    )

    content = _parse_json(content)

    if not content.startswith("{"):
        logger.warning(f"[LLM_CLASSIFY] Could not parse response: {content[:200]}")
        if rich:
            mapping = {v: _null_rich_entry(reason, score) for v in values}
        else:
            mapping = {v: None for v in values}
    else:
        mapping = json.loads(content)

    if rich:
        null_count = sum(1 for v in mapping.values() if isinstance(v, dict) and v.get("value") is None)
    else:
        null_count = sum(1 for v in mapping.values() if v is None)
    logger.info(
        f"[LLM_CLASSIFY] Classified {len(values) - null_count}/{len(values)} items into {len(categories)} categories"
    )

    _notify(LLMCallEvent(
        primitive="llm_classify",
        input_count=len(values),
        null_count=null_count,
        model_used=model_used,
        provider_used=provider_used,
    ))

    return mapping


def llm_extract(texts: str | list[str], fields: list[str] | dict, context: str = "") -> dict[str, str] | list[dict[str, str]]:
    """Extract structured fields from free text using LLM knowledge.

    Args:
        texts: Text string or list of strings to extract from.
        fields: List of field names (or dict whose keys are field names).
        context: Optional context describing the domain.

    Returns:
        Single dict if one text is passed, otherwise list of dicts.
        Each dict maps field names to extracted values (or None).
    """
    # Accept bare string for texts — LLMs often pass doc_read() result directly
    if isinstance(texts, str):
        texts = [texts]
    # Accept dict fields arg (use keys) — LLMs often pass {name: description} dicts
    if isinstance(fields, dict):
        fields = list(fields.keys())
    texts_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    fields_str = ", ".join(f'"{f}"' for f in fields)
    ctx = f" ({context})" if context else ""

    prompt = f"""Extract the following fields from each text{ctx}: {fields_str}.

{texts_str}

Respond with ONLY valid JSON: an array of objects, one per text, each with keys {fields_str}.
If a field cannot be extracted, set it to null.

Example format: [{{"field1": "val1", "field2": null}}, ...]

YOUR JSON RESPONSE:"""

    content, model_used, provider_used = _execute(
        system=f"You extract structured fields from text. Fields: {fields_str}. Output ONLY valid JSON array. Use null for missing.",
        user_message=prompt,
    )

    content = _parse_json(content)

    if not content.startswith("["):
        logger.warning(f"[LLM_EXTRACT] Could not parse response: {content[:200]}")
        results = [{f: None for f in fields} for _ in texts]
    else:
        results = json.loads(content)

    null_count = sum(1 for row in results for v in row.values() if v is None)
    logger.info(
        f"[LLM_EXTRACT] Extracted {len(fields)} fields from {len(texts)} texts"
    )

    _notify(LLMCallEvent(
        primitive="llm_extract",
        input_count=len(texts),
        null_count=null_count,
        model_used=model_used,
        provider_used=provider_used,
    ))

    # Single-text convenience: return dict directly instead of list[dict]
    if len(texts) == 1:
        # noinspection PyTypeChecker
        return results[0]

    return results


def llm_summarize(texts: list[str], instruction: str = "Summarize concisely") -> list[str]:
    """Summarize texts using LLM.

    Args:
        texts: List of texts to summarize.
        instruction: Summarization instruction.

    Returns:
        List of summary strings, one per input text.
    """
    texts_str = "\n---\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))

    prompt = f"""Summarize each of the following {len(texts)} texts. Instruction: {instruction}

{texts_str}

Respond with ONLY valid JSON: an array of strings, one summary per input text.

Example format: ["summary1", "summary2", ...]

YOUR JSON RESPONSE:"""

    content, model_used, provider_used = _execute(
        system=f"You summarize texts. Output ONLY a valid JSON array of strings.",
        user_message=prompt,
    )

    content = _parse_json(content)

    if not content.startswith("["):
        logger.warning(f"[LLM_SUMMARIZE] Could not parse response: {content[:200]}")
        results = ["" for _ in texts]
    else:
        results = json.loads(content)

    null_count = sum(1 for s in results if not s)
    logger.info(
        f"[LLM_SUMMARIZE] Summarized {len(texts)} texts"
    )

    _notify(LLMCallEvent(
        primitive="llm_summarize",
        input_count=len(texts),
        null_count=null_count,
        model_used=model_used,
        provider_used=provider_used,
    ))

    return results


def llm_score(
    texts: list[str],
    min_val: float = 0.0,
    max_val: float = 1.0,
    instruction: str = "Rate each text",
) -> list[tuple[float | None, str]]:
    """Score texts on a numeric scale using LLM judgment.

    Args:
        texts: List of texts to score.
        min_val: Minimum score value (inclusive).
        max_val: Maximum score value (inclusive).
        instruction: Scoring instruction describing what to evaluate.

    Returns:
        List of (score, reasoning) tuples, one per input text.
        score is a float in [min_val, max_val], or None if unscorable.
        reasoning is a short explanation of the score.
    """
    texts_str = "\n---\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))

    prompt = f"""{instruction}

Score each of the following {len(texts)} texts on a scale from {min_val} to {max_val}.

{texts_str}

Respond with ONLY valid JSON: an array of objects, one per input text.
Each object must have "score" (number between {min_val} and {max_val}, or null) and "reasoning" (brief explanation).

Example format: [{{"score": {min_val}, "reasoning": "..."}}, {{"score": {max_val}, "reasoning": "..."}}, {{"score": null, "reasoning": "cannot evaluate"}}]

YOUR JSON RESPONSE:"""

    content, model_used, provider_used = _execute(
        system=f"You score texts on a scale from {min_val} to {max_val}. Output ONLY a valid JSON array of objects with 'score' and 'reasoning' keys.",
        user_message=prompt,
    )

    content = _parse_json(content)

    if not content.startswith("["):
        logger.warning(f"[LLM_SCORE] Could not parse response: {content[:200]}")
        results = [(None, "") for _ in texts]
    else:
        raw = json.loads(content)
        results = []
        for entry in raw:
            if isinstance(entry, dict):
                score = entry.get("score")
                reasoning = entry.get("reasoning", "")
                if score is not None:
                    score = max(min_val, min(max_val, float(score)))
                results.append((score, reasoning))
            elif isinstance(entry, (int, float)):
                # Fallback: plain number without reasoning
                results.append((max(min_val, min(max_val, float(entry))), ""))
            else:
                results.append((None, ""))

    null_count = sum(1 for s, _ in results if s is None)
    logger.info(
        f"[LLM_SCORE] Scored {len(texts) - null_count}/{len(texts)} texts on [{min_val}, {max_val}]"
    )

    _notify(LLMCallEvent(
        primitive="llm_score",
        input_count=len(texts),
        null_count=null_count,
        model_used=model_used,
        provider_used=provider_used,
    ))

    return results
