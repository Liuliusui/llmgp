#!/usr/bin/env python3
"""Tooling layer: primitives, tree helpers, and LLM-guided operators."""

from ._impl.pset_base import add_basic_primitives
from ._impl.trees import (
    extract_subtree_indices_and_trees,
    extract_subtrees,
    pick_deep_k_slices,
    swap_slices_inplace,
)
from ._impl.operators import (
    best_slice_by_llm,
    mate_llm_biased,
    mate_nonllm_subtree,
    mut_llm_guarded,
)

__all__ = [
    "add_basic_primitives",
    "extract_subtree_indices_and_trees",
    "extract_subtrees",
    "pick_deep_k_slices",
    "swap_slices_inplace",
    "best_slice_by_llm",
    "mate_llm_biased",
    "mate_nonllm_subtree",
    "mut_llm_guarded",
]
