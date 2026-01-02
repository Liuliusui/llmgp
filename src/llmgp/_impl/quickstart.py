#!/usr/bin/env python3
"""Beginner-friendly wrapper that hides GP/DEAP wiring."""
from dataclasses import dataclass
import inspect
import random
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
import time
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from deap import algorithms, base, creator, gp, tools

from .api import DEFAULT_ALPHA, DEFAULT_K_SELECT, build_llm_client, compose_system_prompt, llm_score_branch
from .fitness import eval_with_llm_shaping
from .operators import mate_llm_biased, mut_llm_guarded
from .pset_base import add_basic_primitives


@dataclass
class EasyRunResult:
    pop: list
    log: Any
    hof: tools.HallOfFame
    pset: gp.PrimitiveSetTyped
    scorer: Callable[[str], float]


def _infer_arity(fn: Callable) -> int:
    try:
        return len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return 1


def _add_extra_primitives(
    pset: gp.PrimitiveSetTyped,
    extras: Sequence[Union[Callable[..., Any], Tuple[Any, ...]]],
) -> None:
    """
    Allow beginners to inject custom operators without touching DEAP.

    Accepted specs:
      - callable with type hints on args/return (float assumed if missing)
      - (callable, arg_types: Sequence[type], return_type)
      - (callable, arg_types: Sequence[type], return_type, name: str)
    """
    for idx, spec in enumerate(extras):
        fn = spec
        arg_types: Sequence[type] = ()
        ret_type: type = float
        name: Optional[str] = None

        if isinstance(spec, tuple):
            if len(spec) not in (3, 4):
                raise ValueError("extra_primitives tuples must be (fn, arg_types, return_type[, name])")
            fn = spec[0]
            arg_types = spec[1]
            ret_type = spec[2]
            if len(spec) == 4:
                name = spec[3]
        elif callable(spec):
            sig = inspect.signature(spec)
            params = list(sig.parameters.values())
            arg_types = []
            for p in params:
                arg_types.append(p.annotation if p.annotation is not inspect._empty else float)
            ret_type = sig.return_annotation if sig.return_annotation is not inspect._empty else float
        else:
            raise ValueError(f"Unsupported extra_primitives entry at index {idx}: {spec}")

        name = name or getattr(fn, "__name__", f"custom_{idx}")
        # Zero-arity -> treat as terminal
        if len(arg_types) == 0:
            pset.addTerminal(fn(), ret_type, name=name)
            continue
        pset.addPrimitive(fn, arg_types, ret_type, name=name)


def quick_start(
    prompt: str,
    cost_fn: Callable[[Callable[..., Any]], Any],
    feature_fns: Sequence[Callable[..., float]],
    *,
    state_type: type = object,
    extra_primitives: Optional[Sequence[Union[Callable[..., Any], Tuple[Any, ...]]]] = None,
    extra_terminals: Optional[Sequence[Any]] = None,
    pop_size: int = 50,
    ngen: int = 30,
    seed: int = 42,
    max_depth: int = 5,
    n_threads: int = 4,
    cxpb: float = 0.5,
    mutpb: float = 0.3,
    alpha: float = DEFAULT_ALPHA,
    k_select: int = DEFAULT_K_SELECT,
    timeout_ms: int = 15000,
) -> EasyRunResult:
    """
    Run LLM-guided GP with minimal inputs.

    Users provide:
      - prompt: domain description for LLM scoring
      - cost_fn(pf): returns a scalar cost (lower is better)
      - feature_fns: functions that accept the state and return float features
    Optional beginner-friendly extensions:
      - extra_terminals: constants or zero-arg callables to add as terminals
      - extra_primitives: custom ops as callables or (fn, arg_types, return_type[, name])
    """
    random.seed(seed)
    pset = gp.PrimitiveSetTyped("PF", [state_type], float)
    add_basic_primitives(pset)
    if extra_terminals:
        for idx, term in enumerate(extra_terminals):
            val = term() if callable(term) else term
            pset.addTerminal(val, float if isinstance(val, (int, float)) else type(val), name=f"const_{idx}")
    for idx, fn in enumerate(feature_fns):
        arity = _infer_arity(fn)
        name = getattr(fn, "__name__", f"feat_{idx}")
        if arity == 0:
            pset.addTerminal(fn(), float, name=name)
            continue
        arg_types = [state_type] * arity
        pset.addPrimitive(fn, arg_types, float, name=name)
    pset.renameArguments(ARG0="state")
    if extra_primitives:
        _add_extra_primitives(pset, extra_primitives)
    # Ensure at least one primitive of state_type exists to avoid DEAP generation errors.
    if not pset.primitives.get(state_type):
        pset.addPrimitive(lambda s: s, [state_type], state_type, name="state_id")

    full_prompt = compose_system_prompt(prompt)
    client = build_llm_client(timeout_ms=timeout_ms)
    if client is None or not getattr(client, "available", lambda: False)():
        print("[LLM] disabled: no client or credentials, using 0.0 scores")
    else:
        print(f"[LLM] enabled with {getattr(client, '__class__', type('C', (), {})).__name__}")
    scorer = partial(llm_score_branch, system_prompt=full_prompt, client=client)

    if not hasattr(creator, "FitnessEasy"):
        creator.create("FitnessEasy", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualEasy"):
        creator.create("IndividualEasy", gp.PrimitiveTree, fitness=creator.FitnessEasy)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.IndividualEasy, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", mate_llm_biased, k_select=k_select, scorer=scorer)
    toolbox.register("mutate", mut_llm_guarded, expr=toolbox.expr, pset=pset, k_select=k_select, scorer=scorer)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("evaluate", eval_with_llm_shaping, cost_fn=cost_fn, pset=pset, alpha=alpha, k_select=k_select, scorer=scorer)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: (getattr(ind, "sim_score", None), ind.fitness.values[0]))
    stats.register("min_sim", lambda vs: min((v[0] for v in vs if v[0] is not None), default=None))
    stats.register("max_sim", lambda vs: max((v[0] for v in vs if v[0] is not None), default=None))
    stats.register("avg_sim", lambda vs: (sum(v[0] for v in vs if v[0] is not None) / len([v for v in vs if v[0] is not None])) if any(v[0] is not None for v in vs) else None)
    stats.register("min_fit", lambda vs: min(v[1] for v in vs))
    stats.register("max_fit", lambda vs: max(v[1] for v in vs))
    stats.register("avg_fit", lambda vs: sum(v[1] for v in vs) / len(vs))
    start_time = time.time()
    stats.register("time", lambda vs, st=start_time: time.time() - st)
    with ThreadPool(processes=n_threads) as pool:
        toolbox.register("map", pool.map)
        pop, log = algorithms.eaSimple(
            pop,
            toolbox,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=ngen,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

    return EasyRunResult(pop=pop, log=log, hof=hof, pset=pset, scorer=scorer)


__all__ = ["quick_start", "EasyRunResult"]
