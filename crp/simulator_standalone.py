#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Quick use:
#   1) python -m venv .venv && .\.venv\Scripts\activate
#   2) pip install -e .  (from repo root)
#   3) (optional) set LLM_HOST / LLM_API_KEY / LLM_MODEL for LLM scoring
#   4) python crp/simulator_standalone.py  # runs a small demo with bundled data
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from llmgp import FunctionalSimulator, SimulatorConfig
from crp.bay import Bay
from crp.common import TRUCK_POS, T_STEP, T_PICKUP
from crp.read_data import load_instance_from_dat
from crp.prompt import CRP_SYSTEM_PROMPT

PENALTY = 10000


# Set these via environment when you want LLM scoring; keep blanks to avoid hardcoding secrets.
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
    # 如果未填写，将自动使用纯 GP（LLM 评分为 0），无需报错

# ---- 状态与特征 ----
# CRPState 封装了算法需要的可观测量：
#   bay: 堆场状态对象（crp/bay.py），包含各列高度/优先级等
#   seq: 剩余待检索序列（列表）
#   candidate: 当前评估的目的堆索引（可选）
class CRPState:
    def __init__(self, bay: Bay, seq: List[int], candidate=None):
        self.bay = bay
        self.seq = seq
        self.candidate = candidate

def _dst(s: CRPState) -> int:
    return s.candidate if s.candidate is not None else getattr(s.bay, "last_dst", 0)

def _stack_slice(s: CRPState):
    dst = _dst(s)
    return s.bay.pri[dst][: s.bay.h[dst]]

def feat_hla(s: CRPState): return float(s.bay.h[_dst(s)])  # 目标堆高度
def feat_spa(s: CRPState): return float(s.bay.n_tiers - s.bay.h[_dst(s)])  # 目标堆剩余层数
def feat_toppri(s: CRPState):
    dst = _dst(s)
    return float(s.bay.qlt[dst][s.bay.h[dst]-1]) if s.bay.h[dst] else float("inf")
def feat_cur(s: CRPState): return float(s.seq[0]) if s.seq else 0.0
def feat_ri(s: CRPState):
    seq0 = s.seq[0] if s.seq else float("inf")
    return float(sum(1 for c in _stack_slice(s) if c < seq0))
def feat_avg(s: CRPState):
    arr = _stack_slice(s); return float(sum(arr)/len(arr)) if arr else 0.0

FEATURES: Sequence = (feat_hla, feat_spa, feat_toppri, feat_cur, feat_ri, feat_avg)

# ---- pf 适配器：pf(CRPState) -> score ----
def make_pf_adapter(pf):
    def adapter(bay: Bay, seq: List[int], src_stack: int, dst_stack: int):
        bay.last_dst = dst_stack
        bay.current_stack = src_stack
        return float(pf(CRPState(bay=bay, seq=list(seq), candidate=dst_stack)))
    return adapter

# ---- 基础动作 ----
def move_block(bay: Bay, src: int, dst: int):
    bay.move_crane_and_pick(src, dst)
    moving = bay.pri[src][bay.h[src]-1]
    bay.pri[src][bay.h[src]-1] = None
    bay.h[src] -= 1
    new_tier = bay.h[dst]
    bay.pri[dst][new_tier] = moving
    bay.h[dst] += 1
    bay.qlt[dst][new_tier] = moving if new_tier == 0 else min(bay.qlt[dst][new_tier-1], moving)

def retrieve_target(bay: Bay, stack_idx: int):
    bay.retrieve_from(stack_idx)
    top = bay.h[stack_idx]-1
    bay.pri[stack_idx][top] = None
    bay.h[stack_idx] -= 1

# ---- 单实例模拟（支持 RE/REN/UN）----
def run_instance(layout, seq, scheme, pf_adapter, max_steps=50000) -> Tuple[float, float]:
    bay = Bay(len(layout), sum(len(col) for col in layout), deepcopy(layout))
    rehandles = steps = 0
    seq = list(seq)

    for idx, target in enumerate(seq):
        # 找目标列
        s_target = None
        for s in range(bay.n_stacks):
            for t in range(bay.h[s]):
                if bay.pri[s][t] == target:
                    s_target = s; break
            if s_target is not None: break
        if s_target is None:
            continue  # 可能已被提前取走

        # REN 禁止下一目标所在列
        forbidden = set()
        if scheme == "REN" and idx + 1 < len(seq):
            next_target = seq[idx + 1]
            for s in range(bay.n_stacks):
                for t in range(bay.h[s]):
                    if bay.pri[s][t] == next_target:
                        forbidden.add(s); break

        # 清障
        while bay.h[s_target] > 0 and bay.pri[s_target][bay.h[s_target]-1] != target:
            cands = [j for j in range(bay.n_stacks) if j != s_target and bay.h[j] < bay.n_tiers and j not in forbidden]
            if not cands:
                return PENALTY, float(bay.crane_time)
            scores = [(pf_adapter(bay, seq[idx:], s_target, j), j) for j in cands]
            _, dst = min(scores, key=lambda x: x[0])
            move_block(bay, s_target, dst)
            rehandles += 1; steps += 1
            if steps > max_steps:
                return PENALTY, float(bay.crane_time)

        retrieve_target(bay, s_target)
        if steps > max_steps:
            return PENALTY, float(bay.crane_time)

    return float(rehandles), float(bay.crane_time)

# ---- 数据加载（默认用 crp/clean 里所有 .dat）----
def load_crp_instances(data_dir="crp/clean"):
    base = Path(data_dir)
    paths = sorted(base.glob("*.dat"))
    if not paths:
        raise FileNotFoundError(f"No .dat files in {base}")
    return [load_instance_from_dat(str(p)) for p in paths]

# ---- 构建 FunctionalSimulator ----
def build_crp_simulator(
    data_dir="crp/clean",
    scheme="RE",
    prompt: Optional[str] = None,
) -> FunctionalSimulator:
    instances = load_crp_instances(data_dir)

    def data_loader():
        # 仅用于类型推断：返回 (CRPState, target)，target 不用时填 0.0
        states = []
        for layout, seq in instances:
            bay = Bay(len(layout), sum(len(col) for col in layout), deepcopy(layout))
            states.append((CRPState(bay, list(seq)), 0.0))
        return states

    def cost_fn(pf):
        pf_adapter = make_pf_adapter(pf)
        totals = []
        times = []
        for layout, seq in instances:
            rehandles, crane_time = run_instance(layout, seq, scheme, pf_adapter)
            totals.append(rehandles)
            times.append(crane_time)
        return float(sum(totals)), float(sum(times))  # 返回 (重排次数总和, 行车时间总和)

    prompt_text = CRP_SYSTEM_PROMPT
    if prompt:
        prompt_text = f"{CRP_SYSTEM_PROMPT.strip()}\n\nUser prompt:\n{prompt.strip()}\n"
    cfg = SimulatorConfig(prompt=prompt_text, pop_size=3, ngen=50, n_threads=10)
    return FunctionalSimulator(
        data_loader=data_loader,
        feature_fns=FEATURES,
        cost_fn=cost_fn,
        config=cfg,
    )

if __name__ == "__main__":
    _ensure_llm_env()
    sim = build_crp_simulator(data_dir="crp/clean", scheme="RE")  # scheme 可改 RE/REN/UN
    res = sim.run()
    best_pf = sim.best_pf()
    print("Best individual:", res.hof[0], "raw cost:", res.hof[0].sim_score)
