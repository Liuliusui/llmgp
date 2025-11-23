#!/usr/bin/env python3
"""Fitness shaping with LLM guidance."""
from typing import List, Tuple

import numpy as np
from deap import gp

from crp.simulate import apply_relocation_scheme

from .api import DEFAULT_ALPHA, DEFAULT_K_SELECT, llm_score_branch
from .trees import extract_subtrees


def eval_with_llm_shaping(
    individual,
    instances: List[Tuple],
    scheme: str,
    pset,
    alpha: float = DEFAULT_ALPHA,
    k_select: int = DEFAULT_K_SELECT,
    max_steps: int = 5000,
    scorer=llm_score_branch,
):
    pf = gp.compile(expr=individual, pset=pset)
    result = apply_relocation_scheme(instances, scheme, pf, max_steps=max_steps)
    sim_cost = result[0] if isinstance(result, (tuple, list)) else result
    deepest = sorted(extract_subtrees(individual), key=lambda t: t.height, reverse=True)[:k_select]
    imp_scores = [scorer(str(t)) for t in deepest] if deepest else [0.0]
    shaped = sim_cost * (1 - alpha * np.mean(imp_scores))
    individual.sim_score = sim_cost
    return (shaped,)


__all__ = ["eval_with_llm_shaping"]
