#!/usr/bin/env python3
"""LLM (HttpsApi) setup and subtree scoring utilities with safe fallbacks."""
import os
import re
import traceback
from functools import lru_cache
from typing import Optional

from .llm_api import LLMClient

DEFAULT_ALPHA = 0.2
DEFAULT_K_SELECT = 3
LLM_HOST_FALLBACK = ""
LLM_API_KEY_FALLBACK = ""
LLM_MODEL_FALLBACK = ""

BASE_SYSTEM_PROMPT = """
You are an expert at evaluating small Python expression trees ("subtrees") that act as heuristics inside optimization or simulation loops.
Each subtree is a pure Python expression built from a provided GP primitive set and is used to compute a priority or score; the best subtree should improve the downstream objective.

Your task: given a subtree string, output a single float between 0.0 and 1.0 that reflects how promising it is. Higher = more promising. Reply with only the number.

General criteria (apply alongside any domain-specific notes that may be appended below):
  1. Relevance: does the expression use informative features from the primitive set?
  2. Discriminativeness: will it vary meaningfully across different states?
  3. Simplicity: smaller/cleaner expressions are preferred if still useful.
  4. Robustness: avoid pathological cases (divide-by-zero, constants that ignore inputs).
  5. Balance: depth is useful but over-complex trees are penalized.

Formatting: return only a numeric score like `0.73` or `1.0` using a dot for decimals.

If domain-specific guidance is provided, combine it with the rules above and follow it strictly.
"""
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT  # backward-compat alias


def compose_system_prompt(domain_prompt: Optional[str] = None, base_prompt: str = BASE_SYSTEM_PROMPT) -> str:
    """
    Combine the generic scoring prompt with an optional domain-specific prompt.
    """
    if not domain_prompt:
        return base_prompt
    return f"{base_prompt.strip()}\n\nDomain context:\n{domain_prompt.strip()}\n"


def build_llm_client(
    host: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout_ms: int = 5000,
    **kwargs,
) -> Optional[LLMClient]:
    """
    Factory for HttpsApi client. If credentials are incomplete, returns None so callers can safely skip LLM calls.
    """
    host = host or os.getenv("LLM_HOST") or LLM_HOST_FALLBACK
    api_key = api_key or os.getenv("LLM_API_KEY") or LLM_API_KEY_FALLBACK
    model = model or os.getenv("LLM_MODEL") or LLM_MODEL_FALLBACK
    if not (host and api_key and model):
        return None
    try:
        return LLMClient(host, api_key, model, timeout_ms=timeout_ms, **kwargs)
    except Exception as exc:
        print(f"[LLM INIT ERROR] {exc}")
        return None


def _raw_score(branch_code: str, client: LLMClient, system_prompt: str) -> float:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": branch_code},
    ]
    try:
        resp = client.chat(msgs)
    except Exception as exc:
        print(f"[LLM ERROR] {exc}{traceback.format_exc()}")
        return 0.0
    m = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", resp or "")
    if m:
        return float(m.group(1))
    print(f"[LLM WARNING] no numeric score in response: {resp}")
    return 0.0


def llm_score_branch(
    branch_code: str,
    client: Optional[LLMClient] = None,
    system_prompt: Optional[str] = None,
) -> float:
    """
    Score a subtree string via HttpsApi. If no client is configured, returns 0.0 without raising.
    system_prompt can be overridden per task (defaults to SYSTEM_PROMPT).
    """
    system_prompt = system_prompt or SYSTEM_PROMPT
    client = client or _DEFAULT_LLM
    if client is None or not getattr(client, "available", lambda: False)():
        return 0.0
    if client is _DEFAULT_LLM:
        return _cached_score(branch_code, system_prompt)
    return _raw_score(branch_code, client, system_prompt)


@lru_cache(maxsize=5000)
def _cached_score(branch_code: str, system_prompt: str) -> float:
    if _DEFAULT_LLM is None or not _DEFAULT_LLM.available():
        return 0.0
    return _raw_score(branch_code, _DEFAULT_LLM, system_prompt)


# Default client built from environment only (no hardcoded credentials to avoid unwanted calls/timeouts).
_DEFAULT_LLM = build_llm_client()


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_K_SELECT",
    "BASE_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "compose_system_prompt",
    "build_llm_client",
    "llm_score_branch",
]
