#!/usr/bin/env python3
"""GP tree helpers."""
from deap import gp


def swap_slices_inplace(ind1, sl1, ind2, sl2):
    """In-place swap of gp.PrimitiveTree slices."""
    ind1[sl1], ind2[sl2] = ind2[sl2], ind1[sl1]
    return ind1, ind2


def pick_deep_k_slices(ind, k: int = 3):
    """Return the k deepest slices as (depth, start_idx, slice_obj)."""
    triples = []
    for i in range(len(ind)):
        sl = ind.searchSubtree(i)
        depth = gp.PrimitiveTree(ind[sl]).height
        triples.append((depth, i, sl))
    triples.sort(reverse=True, key=lambda x: x[0])
    return triples[:k]


def extract_subtrees(ind):
    subs = []
    for i in range(len(ind)):
        sl = ind.searchSubtree(i)
        nodes = ind[sl]
        subs.append(gp.PrimitiveTree(nodes))
    return subs


def extract_subtree_indices_and_trees(ind):
    result = []
    for i in range(len(ind)):
        sl = ind.searchSubtree(i)
        nodes = ind[sl]
        result.append((i, gp.PrimitiveTree(nodes)))
    return result


__all__ = [
    "swap_slices_inplace",
    "pick_deep_k_slices",
    "extract_subtrees",
    "extract_subtree_indices_and_trees",
]
