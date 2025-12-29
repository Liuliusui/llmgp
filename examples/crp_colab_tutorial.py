# -*- coding: utf-8 -*-
"""
CRP standalone - Colab 手把手教程

用法（在 Colab 中按单元分步运行）：
1) 克隆仓库
2) （可选）安装依赖 / 设置 LLM 环境变量
3) 加载 CRP 数据
4) 定义状态/特征/代价函数
5) 构建并运行 FunctionalSimulator（快速演示）
6) 查看最佳规则 & 可选手动调用
"""

# ========== 1) 克隆仓库 ==========
# 在 Colab 里取消下两行的注释运行
# !rm -rf llmgp
# !git clone https://github.com/Liuliusui/llmgp.git

from __future__ import annotations
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Sequence

import numpy as np

# ========== 2) 加载路径 ==========
# 如果你改了克隆路径，请同步修改 ROOT
ROOT = Path("/content/llmgp") if Path("/content/llmgp").exists() else Path(".").resolve()
sys.path.append(str(ROOT))


# 如需 LLM 评分，请在 Colab 单元里用 %env 或 os.environ 设置 LLM_HOST / LLM_API_KEY / LLM_MODEL
LLM_TIMEOUT_MS = "15000"

# ========== 3) 导入 CRP/LLMGP 核心模块 ==========
from crp.bay import Bay
from crp.read_data import load_instance_from_dat
from crp.prompt import CRP_SYSTEM_PROMPT
from crp.simulator_standalone import run_instance  # 已实现的单实例模拟逻辑
from llmgp import FunctionalSimulator, SimulatorConfig

# ========== 4) 加载 CRP 数据 (.dat) ==========
data_dir = ROOT / "crp" / "clean"
paths = sorted(data_dir.glob("*.dat"))
assert paths, f"No .dat files in {data_dir}"
instances = [load_instance_from_dat(str(p)) for p in paths]

# ========== 5) 定义 State 和特征 ==========
# CRPState: bay(堆场)、seq(剩余目标序列)、candidate(当前目标堆)
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


# 你可以替换/扩展这些特征；下列 6 个是最小可用示例
def feat_hla(s: CRPState): return float(s.bay.h[_dst(s)])                  # 目标堆高度
def feat_spa(s: CRPState): return float(s.bay.n_tiers - s.bay.h[_dst(s)])  # 剩余层数
def feat_toppri(s: CRPState):
    dst = _dst(s)
    return float(s.bay.qlt[dst][s.bay.h[dst]-1]) if s.bay.h[dst] else float("inf")
def feat_cur(s: CRPState): return float(s.seq[0]) if s.seq else 0.0        # 当前目标
def feat_ri(s: CRPState):
    seq0 = s.seq[0] if s.seq else float("inf")
    return float(sum(1 for c in _stack_slice(s) if c < seq0))               # 阻挡数量
def feat_avg(s: CRPState):
    arr = _stack_slice(s); return float(sum(arr)/len(arr)) if arr else 0.0  # 平均优先级


FEATURES: Sequence = (feat_hla, feat_spa, feat_toppri, feat_cur, feat_ri, feat_avg)


# ========== 6) pf 适配器：pf(CRPState) -> score ==========
def make_pf_adapter(pf):
    def adapter(bay: Bay, seq: List[int], src_stack: int, dst_stack: int):
        bay.last_dst = dst_stack
        bay.current_stack = src_stack
        # 使用位置参数以避免笔记本里潜在的命名冲突
        return float(pf(CRPState(bay, list(seq), dst_stack)))
    return adapter


# ========== 7) data_loader / cost_fn ==========
scheme = "RE"  # 可改 "REN" / "UN"


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
    return float(sum(totals)), float(sum(times))  # (重排次数总和, 行车时间总和)


# ========== 8) 配置/构建 FunctionalSimulator ==========
# 如需 LLM 评分，在 Colab 设置环境变量：LLM_HOST / LLM_API_KEY / LLM_MODEL
prompt_user = "CRP relocation heuristic search; keep stacks well-ordered."
cfg = SimulatorConfig(
    prompt=f"{CRP_SYSTEM_PROMPT.strip()}\n\nUser prompt:\n{prompt_user}",
    pop_size=6,   # 为 Colab 缩小规模；想要更好结果可调大
    ngen=5,
    n_threads=1,
)

sim = FunctionalSimulator(
    data_loader=data_loader,
    feature_fns=FEATURES,
    cost_fn=cost_fn,
    config=cfg,
)

# ========== 9) 运行一个快速搜索 ==========
_ensure_llm_env()
print(f"[RUN] CRP scheme={scheme}, pop_size={sim.config.pop_size}, ngen={sim.config.ngen}")
result = sim.run()
best = sim.best_pf()
print("Best individual:", result.hof[0])
print("Raw cost (rehandles, crane_time):", getattr(result.hof[0], "sim_score", None))

# ========== 10) 手动调用最佳规则（可选） ==========
def demo_call():
    class MockBay:
        def __init__(self):
            self.h = [1, 0]
            self.pri = [[1, None], [None, None]]
            self.n_tiers = 2
            self.last_dst = 0
    mock_state = type("S", (), {"bay": MockBay(), "seq": [1], "candidate": 0})
    return float(best(mock_state))

print("Demo call on toy state (priority score):", demo_call())
