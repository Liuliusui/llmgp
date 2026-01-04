#!/usr/bin/env python3
"""Generic GP runner that hides DEAP wiring for multi-arg psets."""
import random
from dataclasses import dataclass
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from typing import Any, Callable, Optional

from deap import algorithms, base, creator, gp, tools

from .api import DEFAULT_ALPHA, DEFAULT_K_SELECT, build_llm_client, compose_system_prompt, llm_score_branch
from .quickstart import EasyRunResult
from .operators import mate_llm_biased, mut_llm_guarded
from .fitness import eval_with_llm_shaping


def _ensure_creator(name: str, base_cls, **kwargs):
    if not hasattr(creator, name):
        creator.create(name, base_cls, **kwargs)


def run_gp_simple(
    *,
    pset: gp.PrimitiveSetTyped,
    cost_fn: Callable[[Callable[..., Any]], Any],
    prompt: str = "",
    pop_size: int = 50,
    ngen: int = 30,
    seed: int = 42,
    min_depth: int = 1,
    max_depth: int = 5,
    n_threads: int = 4,
    cxpb: float = 0.5,
    mutpb: float = 0.3,
    alpha: float = DEFAULT_ALPHA,
    k_select: int = DEFAULT_K_SELECT,
    timeout_ms: int = 15000,
    creator_suffix: str = "Task",
) -> EasyRunResult:
    """
    Generic typed-GP runner that hides all DEAP wiring. Provide a ready pset and cost_fn.
    Returns EasyRunResult(pop, log, hof, pset, scorer).
    """
    random.seed(seed)
    full_prompt = compose_system_prompt(prompt)
    client = build_llm_client(timeout_ms=timeout_ms)
    scorer = partial(llm_score_branch, system_prompt=full_prompt, client=client)

    fit_name = f"Fitness{creator_suffix}"
    ind_name = f"Individual{creator_suffix}"
    _ensure_creator(fit_name, base.Fitness, weights=(-1.0,))
    _ensure_creator(ind_name, gp.PrimitiveTree, fitness=getattr(creator, fit_name))

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genGrow, pset=pset, min_=min_depth, max_=max_depth)
    toolbox.register("individual", tools.initIterate, getattr(creator, ind_name), toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", mate_llm_biased, k_select=k_select, scorer=scorer)
    toolbox.register("mutate", mut_llm_guarded, expr=toolbox.expr, pset=pset, k_select=k_select, scorer=scorer)
    toolbox.register(
        "evaluate",
        eval_with_llm_shaping,
        cost_fn=cost_fn,
        pset=pset,
        alpha=alpha,
        k_select=k_select,
        scorer=scorer,
    )

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(5)
    with ThreadPool(processes=n_threads) as pool:
        toolbox.register("map", pool.map)
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=None, halloffame=hof, verbose=True)

    return EasyRunResult(pop=pop, log=log, hof=hof, pset=pset, scorer=scorer)


__all__ = ["run_gp_simple"]
