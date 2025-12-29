#!/usr/bin/env python3
"""
Standalone YCS simulator runner (no dependency on ycs/gp_simulation.py).

It wires the YCS simulation logic directly into FunctionalSimulator from llmgp:
  - feature_fns expect a dict of terminal values (r_i, p_i, s_ji, d_i, w_i, t, n, bar_r, bar_p, bar_s, bar_d, bar_w)
  - cost_fn runs the simulation and returns total weighted tardiness (lower is better)

Replace `load_instances()` with your real task/time data to run on your cases.
"""
import os
import sys
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple



ROOT = (os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
SRC = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)
from llmgp import SimulatorRunner, SimulatorConfig
from ycs.instance_generator_loader import read_instances_from_folder, read_instances_from_folder_old

# Set these via environment in real runs; keep blanks to avoid hardcoding secrets.
LLM_HOST = ""
LLM_API_KEY = ""
LLM_MODEL = ""
LLM_TIMEOUT_MS = "15000"

def _ensure_llm_env():
    if LLM_HOST and LLM_API_KEY and LLM_MODEL:
        os.environ["LLM_HOST"] = LLM_HOST
        os.environ["LLM_API_KEY"] = LLM_API_KEY
        os.environ["LLM_MODEL"] = LLM_MODEL
        os.environ["LLM_TIMEOUT_MS"] = LLM_TIMEOUT_MS
    # if not set, llm_score_branch will default to 0.0 safely

# ---- Feature functions: state is the terminals dict produced inside simulation ----
def _safe_get(s: Any, key: str):
    # Guard against ill-typed trees: if state is not a dict, just return it as float.
    try:
        return float(s[key])
    except Exception:
        return float(s)

def feat_r_i(s: Dict[str, Any]): return _safe_get(s, "r_i")
def feat_p_i(s: Dict[str, Any]): return _safe_get(s, "p_i")
def feat_s_ji(s: Dict[str, Any]): return _safe_get(s, "s_ji")
def feat_d_i(s: Dict[str, Any]): return _safe_get(s, "d_i")
def feat_w_i(s: Dict[str, Any]): return _safe_get(s, "w_i")
def feat_t(s: Dict[str, Any]): return _safe_get(s, "t")
def feat_n(s: Dict[str, Any]): return _safe_get(s, "n")
def feat_bar_r(s: Dict[str, Any]): return _safe_get(s, "bar_r")
def feat_bar_p(s: Dict[str, Any]): return _safe_get(s, "bar_p")
def feat_bar_s(s: Dict[str, Any]): return _safe_get(s, "bar_s")
def feat_bar_d(s: Dict[str, Any]): return _safe_get(s, "bar_d")
def feat_bar_w(s: Dict[str, Any]): return _safe_get(s, "bar_w")

FEATURES = (
    feat_r_i, feat_p_i, feat_s_ji, feat_d_i, feat_w_i,
    feat_t, feat_n, feat_bar_r, feat_bar_p, feat_bar_s, feat_bar_d, feat_bar_w,
)

epsilon = 1e-7


# ---- YCS simulation logic (no external imports) ----
def min_lookahead(available_task_ids, task_map, time_adj, current_time, prev_id):
    cands = []
    for tid in available_task_ids:
        task = task_map[tid]
        if task["is_internal"]:
            if task["release_time"] < current_time + time_adj[prev_id][tid] + epsilon:
                cands.append(tid)
        else:
            if task["release_time"] < current_time + epsilon:
                cands.append(tid)
    return cands


def dyn_lookahead(available_task_ids, task_map, time_adj, current_time, prev_id):
    cands = min_lookahead(available_task_ids, task_map, time_adj, current_time, prev_id)
    if not cands:
        return cands
    extra = []
    for tid in available_task_ids:
        if tid in cands:
            continue
        rel_t = task_map[tid]["release_time"]
        ok = True
        for cid in cands:
            completion = max(current_time, task_map[cid]["release_time"]) + time_adj[prev_id][cid] + task_map[cid]["processing_time"]
            if rel_t > completion + time_adj[cid][tid] + epsilon:
                ok = False
                break
        if ok:
            extra.append(tid)
    cands.extend(extra)
    return cands


def build_terminals(next_id, cand_ids, task_map, time_adj, current_time, prev_id, cache=None):
    arg = cache or {}
    arg["r_i"] = task_map[next_id]["release_time"]
    arg["p_i"] = task_map[next_id]["processing_time"]
    arg["s_ji"] = time_adj[prev_id][next_id]
    arg["d_i"] = task_map[next_id]["due_date"]
    arg["w_i"] = task_map[next_id]["tardiness_weight"]
    if "t" not in arg: arg["t"] = current_time
    if "n" not in arg: arg["n"] = len(cand_ids)
    if "bar_r" not in arg: arg["bar_r"] = mean(task_map[tid]["release_time"] for tid in cand_ids)
    if "bar_p" not in arg: arg["bar_p"] = mean(task_map[tid]["processing_time"] for tid in cand_ids)
    if "bar_s" not in arg: arg["bar_s"] = mean(time_adj[prev_id][tid] for tid in cand_ids)
    if "bar_d" not in arg: arg["bar_d"] = mean(task_map[tid]["due_date"] for tid in cand_ids)
    if "bar_w" not in arg: arg["bar_w"] = mean(task_map[tid]["tardiness_weight"] for tid in cand_ids)
    return arg


def calc_weighted_tardiness(current_time, prev_id, task_map, time_adj, next_id):
    s_ij = time_adj[prev_id][next_id]
    r_j = task_map[next_id]["release_time"]
    p_j = task_map[next_id]["processing_time"]
    d_j = task_map[next_id]["due_date"]
    if task_map[next_id]["is_internal"]:
        c_j = max(current_time + s_ij, r_j) + p_j
    else:
        c_j = max(current_time, r_j) + s_ij + p_j
    tard = max(0, c_j - d_j)
    return c_j, tard * task_map[next_id]["tardiness_weight"]


def calc_twt_with_seq(task_map, time_adj, seq):
    t = 0
    prev = 0
    twt = 0
    for j in seq:
        t, wt = calc_weighted_tardiness(t, prev, task_map, time_adj, j)
        twt += wt
        prev = j
    return twt


def get_candidates(lookahead, ordered_unscheduled, task_map, time_adj, current_time, prev_id, window=20):
    est_list = []
    for tid, _ in ordered_unscheduled:
        task = task_map[tid]
        est = max(current_time, task["release_time"] - time_adj[prev_id][tid]) if task["is_internal"] else max(current_time, task["release_time"])
        est_list.append((tid, est))
    est_list.sort(key=lambda x: x[1])
    for tid, temp_t in est_list:
        available = []
        for t_id, _ in ordered_unscheduled:
            task = task_map[t_id]
            if task["is_internal"]:
                if task["release_time"] < temp_t + window + epsilon:
                    available.append(t_id)
            else:
                if task["release_time"] < temp_t + epsilon:
                    available.append(t_id)
        cand = lookahead(available, task_map, time_adj, temp_t, prev_id)
        if cand:
            return temp_t, available, cand
    raise RuntimeError("No available tasks")


def simulation_run(task_map, time_adj, lookahead_method, priority_rule):
    ordered = [(tid, t["release_time"]) for tid, t in task_map.items() if tid != 0]
    ordered.sort(key=lambda x: x[1])
    t = 0
    prev = 0
    twt = 0
    seq = [0]
    while ordered:
        t, available, candidates = get_candidates(lookahead_method, ordered, task_map, time_adj, t, prev)
        priorities = {}
        arg_cache = None
        for tid in candidates:
            args = build_terminals(tid, candidates, task_map, time_adj, t, prev, arg_cache)
            arg_cache = args
            priorities[tid] = priority_rule(args)
        next_id = max(priorities, key=priorities.get)
        t, wt = calc_weighted_tardiness(t, prev, task_map, time_adj, next_id)
        twt += wt
        prev = next_id
        seq.append(next_id)
        ordered = [(tid, rt) for tid, rt in ordered if tid != next_id]
    assert len(seq) == len(task_map)
    assert round(calc_twt_with_seq(task_map, time_adj, seq), 2) == round(twt, 2)
    return twt


# ---- Priority rule adapter: pf(state) where state == terminals dict ----
def make_priority_rule(pf: Callable[[Dict[str, float]], float]):
    return lambda terminals: float(pf(terminals))


# ---- Example data loader: replace with your real instances ----
def load_instances(data_dir: str = None) -> List[Tuple[Dict[int, Dict[str, Any]], List[List[float]], float]]:
    """
    Load instances from JSON files, supporting both new and legacy formats.
    Returns list of (task_dict, time_adj_matrix, offline_optimal) where offline_optimal may be None.

    New format per file:
      {"task_dict": {...}, "time_adjacency_matrix": [...], "offline_optimal": <optional>}
    Legacy format per file:
      {"tasks": [...], "time_adjacency_matrix": [...], "minimum_twt": <optional>, "offline_optimal": <optional>}
    """
    base = Path(data_dir or (Path(__file__).parent / "train_instances"))
    if not base.exists():
        raise FileNotFoundError(f"No data_dir found at {base}, please generate instances first.")

    pairs: List[Tuple[Dict[int, Dict[str, Any]], List[List[float]], float]] = []

    # Try new format
    try:
        instances = read_instances_from_folder(str(base))
        for inst in instances:
            task_dict = {int(k): v for k, v in inst["task_dict"].items()}
            best = inst.get("offline_optimal", None)
            pairs.append((task_dict, inst["time_adjacency_matrix"], best))
    except Exception:
        instances = []

    # Fallback to legacy if nothing loaded
    if not pairs:
        legacy = read_instances_from_folder_old(str(base))
        for inst in legacy:
            task_dict = {int(k): v for k, v in inst["task_dict"].items()}
            best = inst.get("offline_optimal") or inst.get("minimum_twt")
            pairs.append((task_dict, inst["time_adjacency_matrix"], best))

    if not pairs:
        raise FileNotFoundError(f"No JSON instances found under {base}")
    return pairs


# ---- Build FunctionalSimulator runner ----
def build_ycs_simulator(instances=None, lookahead="dyn", data_dir=None, prompt: Optional[str] = None):
    instances = instances or load_instances(data_dir=data_dir)
    lookahead_method = dyn_lookahead if lookahead == "dyn" else min_lookahead

    def data_loader():
        # Only used for type inference; real evaluation happens in cost_fn.
        fake_state = {
            "r_i": 0, "p_i": 0, "s_ji": 0, "d_i": 0, "w_i": 1,
            "t": 0, "n": 1, "bar_r": 0, "bar_p": 0, "bar_s": 0, "bar_d": 0, "bar_w": 1,
        }
        return [(fake_state, 0.0)]

    def cost_fn(pf):
        pr = make_priority_rule(pf)
        gaps: List[float] = []
        for idx, (task_map, time_adj, offline_opt) in enumerate(instances):
            obj = simulation_run(task_map, time_adj, lookahead_method, pr)
            if offline_opt and offline_opt > 1e-6:
                # gap percentage relative to offline optimal
                gap = (obj - offline_opt) / offline_opt * 100
            else:
                # no reliable baseline; use raw objective
                gap = obj
            gaps.append(gap)
        return sum(gaps) / len(gaps)

    cfg = SimulatorConfig(
        prompt=prompt or "YCS scheduling",
        pop_size=100,
        ngen=50,
        n_threads=20,
        cxpb=0.5,
        mutpb=0.3,
    )
    return SimulatorRunner(
        data_loader=data_loader,
        feature_fns=FEATURES,
        cost_fn=cost_fn,
        config=cfg,
    )


if __name__ == "__main__":
    _ensure_llm_env()
    sim = build_ycs_simulator(data_dir=Path(__file__).parent / "train_instances")
    print(f"[RUN] pop_size={sim.config.pop_size}, ngen={sim.config.ngen}, cxpb={sim.config.cxpb}, mutpb={sim.config.mutpb}")
    res = sim.run()
    best_pf = sim.best_pf()
    print("Best individual:", res.hof[0], "raw cost:", res.hof[0].sim_score)
