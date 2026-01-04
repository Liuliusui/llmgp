#!/usr/bin/env python3
"""Fitness shaping with LLM guidance (task-agnostic)."""
from typing import Callable, Any

import numpy as np
from deap import gp

from .api import DEFAULT_ALPHA, DEFAULT_K_SELECT, llm_score_branch
from .trees import extract_subtrees


def eval_with_llm_shaping(
    individual,
    cost_fn: Callable[[Callable[..., Any]], Any],
    pset,
    alpha: float = DEFAULT_ALPHA,
    k_select: int = DEFAULT_K_SELECT,
    scorer=llm_score_branch,
):
    """
    Generic fitness shaping: evaluate with cost_fn(compiled_program) and
    scale by LLM-scored subtrees.

    cost_fn should accept the compiled callable (from gp.compile) and
    return either a scalar cost or a tuple/list whose first element is the cost.
    """
    pf = gp.compile(expr=individual, pset=pset)
    result = cost_fn(pf)
    sim_cost = result[0] if isinstance(result, (tuple, list)) else result
    deepest = sorted(extract_subtrees(individual), key=lambda t: t.height, reverse=True)[:k_select]
    imp_scores = [scorer(str(t)) for t in deepest] if deepest else [0.0]
    shaped = sim_cost * (1 - alpha * np.mean(imp_scores))
    individual.sim_score = sim_cost
    return (shaped,)


__all__ = ["eval_with_llm_shaping"]
