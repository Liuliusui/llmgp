#!/usr/bin/env python3
"""
Beginner-friendly simulator template.

Users subclass and implement:
  - load_data(): return list of (state, target)
  - feature_fns(): return sequence of feature functions
Optional override:
  - cost_fn(pf): default uses mean squared error over load_data()
Then call run() to execute quick_start; best_pf() compiles the top individual.
"""
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from deap import gp

from .quickstart import EasyRunResult, quick_start
from .llm import DEFAULT_ALPHA, DEFAULT_K_SELECT


State = Any
FeatureFn = Callable[[State], float]
CostFn = Callable[[Callable[[State], float]], float]
PrimitiveSpec = Union[Callable[..., Any], Tuple[Any, ...]]
TerminalSpec = Any


@dataclass
class SimulatorConfig:
    prompt: str = "Describe your task for LLM scoring (optional)."
    pop_size: int = 10
    ngen: int = 5
    n_threads: int = 1
    seed: int = 42
    max_depth: int = 5
    cxpb: float = 0.5
    mutpb: float = 0.3
    alpha: float = DEFAULT_ALPHA
    k_select: int = DEFAULT_K_SELECT
    timeout_ms: int = 15000


class SimpleSimulator:
    """
    Minimal simulator wrapper: implement load_data + feature_fns, optionally cost_fn.
    Designed for beginners who do not want to touch DEAP wiring.
    """

    def __init__(self, config: Optional[SimulatorConfig] = None):
        self.config = config or SimulatorConfig()
        self.result: Optional[EasyRunResult] = None

    # ---- Override these two to fit your task ----
    def load_data(self) -> List[Tuple[State, float]]:
        """Return list of (state, target) pairs."""
        raise NotImplementedError

    def feature_fns(self) -> Sequence[FeatureFn]:
        """Return feature functions that read from state and output float."""
        raise NotImplementedError

    # ---- Optional override if you need custom cost ----
    def cost_fn(self, pf: Callable[[State], float]) -> float:
        data = self.load_data()
        return sum((pf(s) - t) ** 2 for s, t in data) / len(data)

    # ---- Optional: expose custom primitives/constants without editing GP internals ----
    def extra_primitives(self) -> Sequence[PrimitiveSpec]:
        """Return extra primitives to add to the primitive set (or empty)."""
        return ()

    def extra_terminals(self) -> Sequence[TerminalSpec]:
        """Return extra terminals/constants to add to the primitive set (or empty)."""
        return ()

    # ---- No need to change below for most users ----
    def run(self) -> EasyRunResult:
        cfg = self.config
        self.result = quick_start(
            prompt=cfg.prompt,
            cost_fn=self.cost_fn,
            feature_fns=self.feature_fns(),
            state_type=self._infer_state_type(),
            extra_primitives=self.extra_primitives(),
            extra_terminals=self.extra_terminals(),
            pop_size=cfg.pop_size,
            ngen=cfg.ngen,
            n_threads=cfg.n_threads,
            seed=cfg.seed,
            max_depth=cfg.max_depth,
            cxpb=cfg.cxpb,
            mutpb=cfg.mutpb,
            alpha=cfg.alpha,
            k_select=cfg.k_select,
            timeout_ms=cfg.timeout_ms,
        )
        return self.result

    def best_pf(self, index: int = 0) -> Callable[[State], float]:
        if not self.result:
            raise ValueError("Call run() first.")
        return gp.compile(expr=self.result.hof[index], pset=self.result.pset)

    def _infer_state_type(self):
        data = self.load_data()
        if not data:
            return object
        sample_state, _ = data[0]
        return sample_state.__class__


class FunctionalSimulator(SimpleSimulator):
    """
    Plug-and-play simulator: pass callables instead of subclassing.

    Example:
        sim = FunctionalSimulator(
            data_loader=my_loader,
            feature_fns=[feat_a, feat_b],
            cost_fn=my_cost,  # optional, defaults to MSE over data_loader
            config=SimulatorConfig(pop_size=20, ngen=10),
        )
        sim.run()
        predictor = sim.best_pf()
    """

    def __init__(
        self,
        *,
        data_loader: Callable[[], List[Tuple[State, float]]],
        feature_fns: Sequence[FeatureFn],
        cost_fn: Optional[CostFn] = None,
        extra_primitives: Optional[Sequence[PrimitiveSpec]] = None,
        extra_terminals: Optional[Sequence[TerminalSpec]] = None,
        config: Optional[SimulatorConfig] = None,
    ):
        super().__init__(config=config)
        self._data_loader = data_loader
        self._feature_fns = feature_fns
        self._user_cost_fn = cost_fn
        self._extra_primitives = extra_primitives
        self._extra_terminals = extra_terminals
        self._cached_data: Optional[List[Tuple[State, float]]] = None

    def load_data(self) -> List[Tuple[State, float]]:
        if self._cached_data is None:
            self._cached_data = list(self._data_loader())
        return self._cached_data

    def feature_fns(self) -> Sequence[FeatureFn]:
        return tuple(self._feature_fns)

    def cost_fn(self, pf: Callable[[State], float]) -> float:
        if self._user_cost_fn is not None:
            return self._user_cost_fn(pf)
        return super().cost_fn(pf)

    def extra_primitives(self) -> Sequence[PrimitiveSpec]:
        return self._extra_primitives or ()

    def extra_terminals(self) -> Sequence[TerminalSpec]:
        return self._extra_terminals or ()


__all__ = ["SimulatorConfig", "SimpleSimulator", "FunctionalSimulator"]
