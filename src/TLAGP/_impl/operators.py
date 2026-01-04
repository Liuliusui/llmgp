#!/usr/bin/env python3
"""LLM-biased crossover and mutation operators."""
import random
from typing import Callable

from deap import gp

from .api import DEFAULT_K_SELECT, llm_score_branch
from .trees import extract_subtree_indices_and_trees, pick_deep_k_slices, swap_slices_inplace


def best_slice_by_llm(ind, k_select: int = DEFAULT_K_SELECT, scorer: Callable[[str], float] = llm_score_branch):
    """Choose the LLM-best slice among the deepest k slices."""
    top = pick_deep_k_slices(ind, k=k_select)
    best = None
    best_score = -1.0
    for _, i, sl in top:
        subtree_str = str(gp.PrimitiveTree(ind[sl]))
        sc = scorer(subtree_str)
        if sc > best_score:
            best_score, best = sc, (i, sl)
    return best  # (start_idx, slice) or None


def mate_llm_biased(p1, p2, k_select: int = DEFAULT_K_SELECT, scorer: Callable[[str], float] = llm_score_branch):
    """Crossover guided by LLM-scored subtrees."""
    best1 = best_slice_by_llm(p1, k_select=k_select, scorer=scorer)
    best2 = best_slice_by_llm(p2, k_select=k_select, scorer=scorer)
    if best1 is None or best2 is None:
        return p1, p2
    _, sl1 = best1
    _, sl2 = best2
    swap_slices_inplace(p1, sl1, p2, sl2)
    return p1, p2


def mate_nonllm_subtree(p1, p2, k_select: int = DEFAULT_K_SELECT):
    """Fallback crossover using deepest slices without LLM scoring."""
    top1 = pick_deep_k_slices(p1, k=k_select)
    top2 = pick_deep_k_slices(p2, k=k_select)
    _, _, sl1 = random.choice(top1)
    _, _, sl2 = random.choice(top2)
    swap_slices_inplace(p1, sl1, p2, sl2)
    return p1, p2


def mut_llm_guarded(
    ind,
    expr,
    pset,
    k_select: int = DEFAULT_K_SELECT,
    mutpb: float = 0.1,
    scorer: Callable[[str], float] = llm_score_branch,
):
    """Mutation guided by LLM scores to preserve promising subtrees."""
    subtrees = extract_subtree_indices_and_trees(ind)
    if not subtrees:
        return (ind,)

    subtrees_with_depth = [(idx, tree, tree.height) for idx, tree in subtrees]
    topk = sorted(subtrees_with_depth, key=lambda x: x[2], reverse=True)[:k_select]
    scored = [(idx, tree, scorer(str(tree))) for idx, tree, _ in topk]
    if not scored:
        return (ind,)

    best_idx, _, best_score = max(scored, key=lambda x: x[2])
    topk_idx = set(idx for idx, _, _ in scored)

    mutate_idxs = []
    if random.random() > best_score:
        mutate_idxs.append(best_idx)
    for idx, _, _ in scored:
        if idx == best_idx:
            continue
        if random.random() < mutpb:
            mutate_idxs.append(idx)
    for idx, _ in subtrees:
        if idx in topk_idx:
            continue
        if random.random() < mutpb:
            mutate_idxs.append(idx)

    mutate_idxs = sorted(set(mutate_idxs), reverse=True)
    for idx in mutate_idxs:
        if idx >= len(ind):
            continue
        sl = ind.searchSubtree(idx)
        ind[sl] = expr()

    return (ind,)


__all__ = [
    "best_slice_by_llm",
    "mate_llm_biased",
    "mate_nonllm_subtree",
    "mut_llm_guarded",
]
