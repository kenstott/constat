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

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
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

    def generate_vision(
        self,
        system: str,
        image_bytes: bytes,
        mime_type: str,
        text_prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """Vision generation for standalone mode."""
        image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")

        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            resp = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": text_prompt},
                    ],
                }],
            )
            return resp.content[0].text

        if self.provider in ("openai", "grok", "together", "groq", "mistral"):
            import openai
            base_urls = {
                "grok": "https://api.x.ai/v1",
                "together": "https://api.together.xyz/v1",
                "groq": "https://api.groq.com/openai/v1",
                "mistral": "https://api.mistral.ai/v1",
            }
            kwargs = {}
            if self.provider in base_urls:
                kwargs["base_url"] = base_urls[self.provider]
            client = openai.OpenAI(api_key=self.api_key, **kwargs)
            data_uri = f"data:{mime_type};base64,{image_b64}"
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": text_prompt},
                    ]},
                ],
            )
            return resp.choices[0].message.content

        raise RuntimeError(f"Vision not supported for provider: {self.provider}")


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

def _execute(system: str, user_message: str, *, task_type=None) -> tuple[str, str, str]:
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
            task_type=task_type or TaskType.SYNTHESIS,
            system=system,
            user_message=user_message,
            max_tokens=backend.max_output_tokens,
        )
        if not result.success:
            raise RuntimeError(result.content)
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


def _execute_vision(
    system: str, image_bytes: bytes, mime_type: str, text_prompt: str,
) -> tuple[str, str, str]:
    """Call the backend with a vision request and return (content, model_used, provider_used)."""
    backend = _get_backend()

    if isinstance(backend, _DirectProvider):
        content = backend.generate_vision(
            system=system, image_bytes=image_bytes,
            mime_type=mime_type, text_prompt=text_prompt,
        )
        return content.strip(), backend.model, backend.provider

    # TaskRouter / BaseLLMProvider — has generate_vision()
    if hasattr(backend, "generate_vision"):
        content = backend.generate_vision(
            system=system, image_bytes=image_bytes,
            mime_type=mime_type, text_prompt=text_prompt,
        )
        return (
            content.strip(),
            getattr(backend, "model", "unknown"),
            type(backend).__name__,
        )

    raise RuntimeError("Backend does not support vision")


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

    from constat.core.models import TaskType
    content, model_used, provider_used = _execute(
        system=system_msg,
        user_message=prompt,
        task_type=TaskType.STRUCTURED_EXTRACTION,
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
            retry_content, _, _ = _execute(system=system_msg, user_message=retry_prompt, task_type=TaskType.STRUCTURED_EXTRACTION)
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

    from constat.core.models import TaskType
    content, model_used, provider_used = _execute(
        system=system_msg,
        user_message=prompt,
        task_type=TaskType.STRUCTURED_EXTRACTION,
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
All values MUST be plain strings — never use arrays, lists, or nested objects as values. Join multiple items with commas.

Example format: [{{"field1": "val1", "field2": null}}, ...]

YOUR JSON RESPONSE:"""

    from constat.core.models import TaskType
    content, model_used, provider_used = _execute(
        system=f"You extract structured fields from text. Fields: {fields_str}. Output ONLY valid JSON array. Use null for missing.",
        user_message=prompt,
        task_type=TaskType.STRUCTURED_EXTRACTION,
    )

    content = _parse_json(content)

    if not content.startswith("["):
        logger.warning(f"[LLM_EXTRACT] Could not parse response: {content[:200]}")
        results = [{f: None for f in fields} for _ in texts]
    else:
        results = json.loads(content)
        # Flatten any array values to comma-separated strings
        for row in results:
            for k, v in row.items():
                if isinstance(v, list):
                    row[k] = ", ".join(str(x) for x in v)

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

    from constat.core.models import TaskType
    content, model_used, provider_used = _execute(
        system=f"You summarize texts. Output ONLY a valid JSON array of strings.",
        user_message=prompt,
        task_type=TaskType.SUMMARIZATION,
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

    from constat.core.models import TaskType
    content, model_used, provider_used = _execute(
        system=f"You score texts on a scale from {min_val} to {max_val}. Output ONLY a valid JSON array of objects with 'score' and 'reasoning' keys.",
        user_message=prompt,
        task_type=TaskType.STRUCTURED_EXTRACTION,
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


_UNIT_MULT = {
    'k': 1e3, 'K': 1e3,
    'm': 1e6, 'M': 1e6,
    'b': 1e9, 'B': 1e9,
    't': 1e12, 'T': 1e12,
}
_CURRENCY_SYMBOLS = frozenset('$£€¥₹₽₩₪฿₫₴₸₺₼₾')
_CURRENCY_RE = re.compile('[' + re.escape(''.join(_CURRENCY_SYMBOLS)) + '%,]')
_BOOL_TRUE = frozenset({'true', 'yes', 'y'})
_BOOL_FALSE = frozenset({'false', 'no', 'n'})
_NULL_VALS = frozenset({'null', 'none', 'n/a', 'na', 'nan', '-', '--', ''})
# Column names that should never be coerced to numeric
_STRING_COL_RE = re.compile(
    r'(?i)(^id$|_id$|id_|name|desc|note|comment|label|title|text|'
    r'address|email|phone|url|path|code|sku|status|type|category)'
)


def _parse_cell_value(val):
    """Parse a cell value into its simplest type.

    Returns:
        bool    — "Yes"/"No", "True"/"False", "Y"/"N"
        int     — whole numbers: "5", "1,000"
        float   — decimals/pct/currency: "8.5%", "$1,200.50", "10k"
        tuple   — ranges: "5-8%", "8 to 12", "up to 15%"
        None    — non-numeric text, dates, IDs, codes
        passthrough — already bool/int/float
    """
    # --- Passthrough ---
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if isinstance(val, float) and val != val:  # NaN
            return None
        return val

    s = str(val).strip()
    if not s or s.lower() in _NULL_VALS:
        return None

    s_lower = s.lower()

    # --- Bool detection ---
    if s_lower in _BOOL_TRUE:
        return True
    if s_lower in _BOOL_FALSE:
        return False

    # --- Bail-out: dates (ISO, US, EU) ---
    if re.match(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}$', s):
        return None
    if re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$', s):
        return None

    # --- Bail-out: times ---
    if re.match(r'\d{1,2}:\d{2}(:\d{2})?$', s):
        return None

    # --- Bail-out: significant alpha content (IDs, codes, prose) ---
    # Remove known range phrases from the original string first (word boundaries intact)
    residual = re.sub(r'(?i)\b(up\s+to|between|and|or|to)\b', '', s)
    # Then strip digits, currency symbols, whitespace, punctuation — see what letters remain
    residual = _CURRENCY_RE.sub('', residual)
    residual = re.sub(r'[\d.\s\-–—/;|()+:]+', '', residual)
    # Allow unit suffix characters (k, M, B, T)
    residual = residual.strip()
    if residual and not all(c in _UNIT_MULT for c in residual):
        return None

    # --- Detect percentage → convert to decimal at the end ---
    is_pct = '%' in s

    # --- Accounting negatives: (5%) → -5 ---
    is_accounting_neg = s.startswith('(') and s.endswith(')')
    if is_accounting_neg:
        s = s[1:-1].strip()

    # Strip currency symbols, %, commas
    cleaned = _CURRENCY_RE.sub('', s).strip()

    def _apply_unit(num_str: str):
        num_str = num_str.strip()
        if not num_str:
            return None
        if num_str[-1] in _UNIT_MULT:
            try:
                return float(num_str[:-1]) * _UNIT_MULT[num_str[-1]]
            except ValueError:
                return None
        try:
            return float(num_str)
        except ValueError:
            return None

    def _pct(v):
        """If original had %, convert to decimal."""
        if not is_pct:
            return v
        if isinstance(v, tuple):
            return tuple(n / 100 for n in v)
        return v / 100

    # --- "up to X" → (0, X) ---
    up_to = re.match(r'up\s+to\s+', s, re.IGNORECASE)
    if up_to:
        rest = _CURRENCY_RE.sub('', s[up_to.end():]).strip()
        v = _apply_unit(rest)
        if v is not None:
            return _pct((0.0, v))

    # --- "between X and Y" → (X, Y) ---
    between = re.match(r'between\s+', s, re.IGNORECASE)
    if between:
        rest = _CURRENCY_RE.sub('', s[between.end():]).strip()
        parts = re.split(r'\s+and\s+', rest, flags=re.IGNORECASE)
        if len(parts) == 2:
            a, b = _apply_unit(parts[0].strip()), _apply_unit(parts[1].strip())
            if a is not None and b is not None:
                return _pct((a, b))

    # --- Range detection ---
    range_parts = re.split(
        r'\s*[-–—/]\s*|\s+to\s+|\s+and\s+',
        cleaned,
        flags=re.IGNORECASE,
    )
    if len(range_parts) > 1:
        numbers = [_apply_unit(p) for p in range_parts if p.strip()]
        numbers = [n for n in numbers if n is not None]
        if len(numbers) >= 2:
            if is_accounting_neg:
                numbers = [-n for n in numbers]
            return _pct(tuple(numbers))

    # --- Single value ---
    v = _apply_unit(cleaned)
    if v is not None:
        if is_accounting_neg:
            v = -v
        v = _pct(v)
        if not is_pct and v == int(v) and not ('.' in cleaned or cleaned[-1] in _UNIT_MULT):
            return int(v)
        return v

    return None


def llm_extract_table(
    text: str,
    description: str,
    columns: list[str] | None = None,
    dtypes: dict[str, str] | None = None,
) -> "pd.DataFrame":
    """Extract a table from document text into a DataFrame.

    Reads structured tabular data (markdown tables, lists of key-value pairs,
    embedded grids) from document text and returns it as a DataFrame.

    Args:
        text: Document text (typically from doc_read()).
        description: What table to find (e.g., "employee rating scale",
                     "raise percentage guidelines").
        columns: Optional column names to enforce. If None, LLM infers them.
        dtypes: Optional dict mapping column names to types for conversion.
                Supported types: 'str', 'int', 'float', 'bool'.
                Example: {'rating': 'str', 'min_raise_rate': 'float'}

    Returns:
        pandas DataFrame with extracted rows and columns.

    Raises:
        ValueError: If no table is found or extraction fails.
    """
    import pandas as pd

    col_instruction = ""
    if columns:
        col_str = ", ".join(f'"{c}"' for c in columns)
        col_instruction = f"\nUse exactly these column names as keys: {col_str}."

    prompt = f"""Locate the table described as: "{description}" in the following text.
Extract ALL rows from that table.

{text}

Return ONLY valid JSON: an array of objects, one per row.{col_instruction}
If no matching table is found, return an empty array [].

YOUR JSON RESPONSE:"""

    from constat.core.models import TaskType
    content, model_used, provider_used = _execute(
        system="You extract tabular data from documents. Output ONLY a valid JSON array of row objects.",
        user_message=prompt,
        task_type=TaskType.STRUCTURED_EXTRACTION,
    )

    content = _parse_json(content)

    if not content.startswith("["):
        logger.warning(f"[LLM_EXTRACT_TABLE] Could not parse response: {content[:200]}")
        raise ValueError(
            f"llm_extract_table failed to extract '{description}': LLM did not return a valid JSON array. "
            f"Try a more specific description or verify the document contains the expected table."
        )
    else:
        rows = json.loads(content)

    if len(rows) == 0:
        raise ValueError(
            f"llm_extract_table found no rows for '{description}'. "
            f"The document may not contain a matching table, or the description may need to be more specific."
        )

    logger.info(
        f"[LLM_EXTRACT_TABLE] Extracted {len(rows)} rows for '{description}'"
    )

    _notify(LLMCallEvent(
        primitive="llm_extract_table",
        input_count=1,
        null_count=0 if rows else 1,
        model_used=model_used,
        provider_used=provider_used,
    ))

    df = pd.DataFrame(rows)

    # --- Phase 1: Coerce ALL columns first (before enforcement) ---
    _pct_cols = set()  # columns where % values were converted to decimal rates
    for col in list(df.columns):
        if _STRING_COL_RE.search(col):
            continue
        if df[col].dtype.kind not in ('O', 'U'):
            continue

        non_null = df[col].notna().sum()
        if non_null == 0:
            continue

        # Detect if column has percentage values
        pct_count = df[col].apply(
            lambda v: '%' in str(v) if pd.notna(v) else False
        ).sum()
        if pct_count > non_null * 0.3:
            _pct_cols.add(col)

        parsed = df[col].apply(
            lambda v: _parse_cell_value(v) if pd.notna(v) else None
        )

        # Bool column
        bool_count = parsed.apply(lambda v: isinstance(v, bool)).sum()
        if bool_count > non_null * 0.5:
            df[col] = parsed.apply(
                lambda v: v if isinstance(v, bool) else None
            ).astype('boolean')
            continue

        # Date column
        date_like = df[col].apply(
            lambda v: bool(re.match(
                r'\d{4}[-/]\d{1,2}[-/]\d{1,2}$|'
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$',
                str(v).strip(),
            )) if pd.notna(v) else False
        ).sum()
        if date_like > non_null * 0.5:
            converted = pd.to_datetime(df[col], errors='coerce')
            if converted.notna().sum() > non_null * 0.5:
                df[col] = converted
                continue

        # Numeric / range column
        numeric_count = parsed.apply(
            lambda v: isinstance(v, (int, float, tuple)) and not isinstance(v, bool)
        ).sum()
        if numeric_count > non_null * 0.5:
            has_ranges = parsed.apply(lambda v: isinstance(v, tuple)).any()
            if has_ranges:
                df[col] = parsed
            else:
                # Use Int64 when all non-null values are whole integers
                all_int = parsed.apply(
                    lambda v: isinstance(v, int) and not isinstance(v, bool)
                    if v is not None else True
                ).all()
                if all_int:
                    df[col] = parsed.apply(
                        lambda v: int(v) if isinstance(v, (int, float))
                        and not isinstance(v, bool) else pd.NA
                    ).astype('Int64')
                else:
                    df[col] = parsed.apply(
                        lambda v: float(v) if isinstance(v, (int, float))
                        and not isinstance(v, bool) else None
                    ).astype(float)

    # --- Phase 1b: Rename pct → rate for percent-converted columns ---
    _PCT_RE = re.compile(r'(?i)(percentage|percent|pct)')
    renames = {}
    for col in _pct_cols:
        if col in df.columns and _PCT_RE.search(col):
            new_name = _PCT_RE.sub('rate', col)
            if new_name != col:
                renames[col] = new_name
    if renames:
        df = df.rename(columns=renames)
        logger.info(f"[LLM_EXTRACT_TABLE] Renamed pct→rate: {renames}")
        # Update _pct_cols to track new names for tuple expansion
        _pct_cols = {renames.get(c, c) for c in _pct_cols}

    # --- Phase 2: Auto-expand tuple columns by width ---
    # 2-element → _min, _max
    # 3-element → _min, _mid, _max
    # 4-element → _q1, _q2, _q3, _q4
    # Ragged   → pad shorter tuples with None to max width
    _SUFFIXES = {
        2: ('_min', '_max'),
        3: ('_min', '_mid', '_max'),
        4: ('_q1', '_q2', '_q3', '_q4'),
    }
    for col in list(df.columns):
        if not df[col].apply(lambda v: isinstance(v, tuple)).any():
            continue
        tuple_lens = df[col].apply(
            lambda v: len(v) if isinstance(v, tuple) else 0
        )
        max_width = int(tuple_lens.max())
        if max_width < 2 or max_width > 4:
            continue  # don't expand single-element or very wide tuples

        suffixes = _SUFFIXES.get(max_width, tuple(f'_{i+1}' for i in range(max_width)))

        def _scalar_to_float(v):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            return None

        for i, suffix in enumerate(suffixes):
            df[f'{col}{suffix}'] = df[col].apply(
                lambda v, idx=i: (
                    v[idx] if isinstance(v, tuple) and idx < len(v)
                    else _scalar_to_float(v) if not isinstance(v, tuple)
                    else None  # tuple shorter than max_width
                )
            )
        df = df.drop(columns=[col])
        expanded = ', '.join(f"'{col}{s}'" for s in suffixes)
        logger.info(
            f"[LLM_EXTRACT_TABLE] Expanded range column '{col}' → {expanded}"
        )

    # --- Phase 3: Enforce requested columns (with fuzzy matching) ---
    if columns:
        # Build mapping: requested col → best available col
        available = set(df.columns)
        col_map = {}
        for req_col in columns:
            if req_col in available:
                col_map[req_col] = req_col
                continue
            # Try suffix match: "min_raise" matches "typical_raise_min"
            req_lower = req_col.lower()
            req_parts = set(req_lower.replace('_', ' ').split())
            _POSITION_KEYWORDS = {'min', 'max', 'mid', 'q1', 'q2', 'q3', 'q4'}
            for avail in available:
                avail_lower = avail.lower()
                avail_parts = set(avail_lower.replace('_', ' ').split())
                # Exact word-set match: "min_raise" ↔ "raise_min"
                if req_parts == avail_parts:
                    col_map[req_col] = avail
                    break
                # Positional: "min_raise" matches "typical_raise_min"
                # if they share a position keyword and a content word
                pos_in_req = req_parts & _POSITION_KEYWORDS
                pos_in_avail = avail_parts & _POSITION_KEYWORDS
                if pos_in_req and pos_in_req == pos_in_avail:
                    content_req = req_parts - _POSITION_KEYWORDS
                    content_avail = avail_parts - _POSITION_KEYWORDS
                    if content_req & content_avail:
                        col_map[req_col] = avail
                        break

        # LLM fallback for unmatched columns — reuse llm_classify
        unmatched_req = [c for c in columns if c not in col_map]
        unmatched_avail = list(available - set(col_map.values()))
        if unmatched_req and unmatched_avail:
            try:
                matches = llm_classify(
                    unmatched_req, unmatched_avail,
                    context="Match requested column names to available column names",
                )
                for req, avail in matches.items():
                    if avail and avail in unmatched_avail:
                        col_map[req] = avail
                        logger.info(
                            f"[LLM_EXTRACT_TABLE] LLM matched '{req}' → '{avail}'"
                        )
            except Exception as e:
                logger.debug(f"[LLM_EXTRACT_TABLE] LLM column matching failed: {e}")

        # Apply mapping
        result_df = pd.DataFrame()
        for req_col in columns:
            if req_col in col_map:
                result_df[req_col] = df[col_map[req_col]]
            else:
                result_df[req_col] = None

        mapped = {r: c for r, c in col_map.items() if r != c}
        if mapped:
            logger.info(f"[LLM_EXTRACT_TABLE] Column mapping: {mapped}")

        extra = available - set(col_map.values())
        if extra:
            logger.warning(
                f"[LLM_EXTRACT_TABLE] Dropping unmatched columns: {extra}"
            )

        df = result_df

    # --- Phase 4: Apply explicit dtype conversions ---
    if dtypes:
        _DTYPE_MAP = {'str': str, 'int': 'Int64', 'float': float, 'bool': 'boolean'}
        for col, dtype_str in dtypes.items():
            if col not in df.columns:
                continue
            target = _DTYPE_MAP.get(dtype_str, dtype_str)
            try:
                if target is str:
                    df[col] = df[col].astype(str).str.strip()
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce') if target in (float, 'Int64') else df[col].astype(target)
                    if target == 'Int64':
                        df[col] = df[col].astype('Int64')
                logger.info(f"[LLM_EXTRACT_TABLE] Converted '{col}' to {dtype_str}")
            except Exception as e:
                logger.warning(f"[LLM_EXTRACT_TABLE] Failed to convert '{col}' to {dtype_str}: {e}")

    return df


def llm_extract_facts(
    text: str,
    context: str = "",
) -> list[dict]:
    """Extract all facts from text with typed metadata.

    Scans document text and identifies every discrete factual assertion,
    data point, rule, or structured element. Each fact is tagged with
    its data type and relevant metadata.

    Args:
        text: Text to scan (from doc_read() or chunk content).
        context: Optional domain context (e.g., "HR compensation policies").

    Returns:
        List of fact dicts. Each has:
        - name: short label for the fact
        - value: the extracted value (string, number, or structured)
        - dtype: "scalar" | "range" | "table" | "list" | "rule" | "text"
        - metadata: type-specific metadata dict
    """
    ctx = f"\nDomain context: {context}" if context else ""

    prompt = f"""Extract ALL discrete facts, data points, rules, thresholds, tables, and lists from the following text.{ctx}

{text}

For each fact, return a JSON object with:
- "name": short label
- "value": the extracted value (string, number, array of objects for tables, array for lists)
- "dtype": one of "scalar", "range", "table", "list", "rule", "text"
- "metadata": type-specific dict:
  - scalar: {{"unit": "...", "data_type": "..."}}
  - range: {{"min": ..., "max": ..., "unit": "..."}}
  - table: {{"columns": [...], "row_count": N}}
  - list: {{"items": [...], "count": N}}
  - rule: {{"condition": "...", "action": "..."}}
  - text: {{}}

Return ONLY valid JSON: an array of fact objects.

YOUR JSON RESPONSE:"""

    from constat.core.models import TaskType
    content, model_used, provider_used = _execute(
        system="You extract structured facts from documents. Output ONLY a valid JSON array of fact objects.",
        user_message=prompt,
        task_type=TaskType.STRUCTURED_EXTRACTION,
    )

    content = _parse_json(content)

    if not content.startswith("["):
        logger.warning(f"[LLM_EXTRACT_FACTS] Could not parse response: {content[:200]}")
        facts = []
    else:
        facts = json.loads(content)

    logger.info(
        f"[LLM_EXTRACT_FACTS] Extracted {len(facts)} facts"
    )

    _notify(LLMCallEvent(
        primitive="llm_extract_facts",
        input_count=1,
        null_count=0 if facts else 1,
        model_used=model_used,
        provider_used=provider_used,
    ))

    return facts


def llm_vision(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    system: str | None = None,
) -> str:
    """Analyze an image using LLM vision capabilities.

    Args:
        image_bytes: Raw image bytes.
        mime_type: Image MIME type (e.g., "image/png", "image/jpeg").
        prompt: Text prompt describing what to analyze.
        system: Optional system prompt.

    Returns:
        Plain text response from the LLM.
    """
    sys_msg = system or "You analyze images. Describe what you see accurately and concisely."

    content, model_used, provider_used = _execute_vision(
        system=sys_msg,
        image_bytes=image_bytes,
        mime_type=mime_type,
        text_prompt=prompt,
    )

    _notify(LLMCallEvent(
        primitive="llm_vision",
        input_count=1,
        null_count=0 if content else 1,
        model_used=model_used,
        provider_used=provider_used,
    ))

    return content


def llm_translate(
    texts: list[str],
    target_language: str,
    source_language: str | None = None,
) -> list[str]:
    """Translate texts to a target language using LLM.

    Args:
        texts: List of texts to translate.
        target_language: Language to translate into (e.g., "English", "French").
        source_language: Optional source language hint.

    Returns:
        List of translated strings, one per input text.
    """
    texts_str = "\n---\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    source_hint = f" from {source_language}" if source_language else ""

    prompt = f"""Translate each of the following {len(texts)} texts{source_hint} to {target_language}.

{texts_str}

Respond with ONLY valid JSON: an array of strings, one translation per input text.
Preserve the original meaning. Do not add explanations.

Example format: ["translation1", "translation2", ...]

YOUR JSON RESPONSE:"""

    from constat.core.models import TaskType
    content, model_used, provider_used = _execute(
        system=f"You translate text to {target_language}. Output ONLY a valid JSON array of strings.",
        user_message=prompt,
        task_type=TaskType.SUMMARIZATION,
    )

    content = _parse_json(content)

    if not content.startswith("["):
        logger.warning(f"[LLM_TRANSLATE] Could not parse response: {content[:200]}")
        results = ["" for _ in texts]
    else:
        results = json.loads(content)

    null_count = sum(1 for s in results if not s)
    logger.info(f"[LLM_TRANSLATE] Translated {len(texts)} texts to {target_language}")

    _notify(LLMCallEvent(
        primitive="llm_translate",
        input_count=len(texts),
        null_count=null_count,
        model_used=model_used,
        provider_used=provider_used,
    ))

    return results
