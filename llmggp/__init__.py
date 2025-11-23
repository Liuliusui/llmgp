#!/usr/bin/env python3
"""
LLM-guided GP reusable toolkit: LLM scoring, GP tree helpers, biased operators,
and fitness shaping. Domain-specific primitive sets/prompts live in task modules.
"""
from .api import (
    DEFAULT_ALPHA,
    DEFAULT_K_SELECT,
    SYSTEM_PROMPT,
    build_llm_client,
    compose_system_prompt,
    llm_score_branch,
)
from .fitness import eval_with_llm_shaping
from .operators import best_slice_by_llm, mate_llm_biased, mate_nonllm_subtree, mut_llm_guarded
from .pset_base import add_basic_primitives
from .trees import extract_subtree_indices_and_trees, extract_subtrees, pick_deep_k_slices, swap_slices_inplace

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_K_SELECT",
    "SYSTEM_PROMPT",
    "compose_system_prompt",
    "build_llm_client",
    "llm_score_branch",
    "eval_with_llm_shaping",
    "best_slice_by_llm",
    "mate_llm_biased",
    "mate_nonllm_subtree",
    "mut_llm_guarded",
    "add_basic_primitives",
    "extract_subtree_indices_and_trees",
    "extract_subtrees",
    "pick_deep_k_slices",
    "swap_slices_inplace",
]
