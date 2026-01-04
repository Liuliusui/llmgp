#!/usr/bin/env python3
"""LLM connectivity and scoring helpers."""

from ._impl.api import (
    DEFAULT_ALPHA,
    DEFAULT_K_SELECT,
    SYSTEM_PROMPT,
    compose_system_prompt,
    build_llm_client,
    llm_score_branch,
)
from ._impl.llm_api import HttpsApi, LLMClient, LLMTransportError

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_K_SELECT",
    "SYSTEM_PROMPT",
    "compose_system_prompt",
    "build_llm_client",
    "llm_score_branch",
    "LLMClient",
    "LLMTransportError",
    "HttpsApi",
]
