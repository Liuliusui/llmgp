# llmgp

LLM-guided genetic programming utilities with beginner-friendly entry points.

## Install (dev)
```
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -e .
```
Requires Python 3.9+, `deap`, `numpy`.

## Quickstart (single-input tasks)
```python
from llmgp import quick_start

class State(dict): ...

def feat_x(s: State) -> float: return float(s.get("x", 0.0))
def feat_y(s: State) -> float: return float(s.get("y", 0.0))

def cost_fn(pf):
    data = [(State(x=1, y=2), 0.0), (State(x=3, y=4), 1.0)]
    return sum((pf(s) - t) ** 2 for s, t in data) / len(data)

result = quick_start(
    prompt="Regress a simple function of x and y.",
    cost_fn=cost_fn,
    feature_fns=[feat_x, feat_y],
    state_type=State,
    pop_size=10,
    ngen=5,
    n_threads=1,
)
print("Top individual:", result.hof[0])
```
Full sample: `examples/basic_regression.py`.

### Extend without touching GP
- Add ops via `extra_primitives=[my_op]` or `(my_op, [float, float], float, "my_op")`
- Add constants via `extra_terminals=[0.5, lambda: 2.0]`

## Generic runner (multi-arg psets)
Already have a custom `pset` and `cost_fn`? Let `gp_run_with_pset` wire DEAP for you:
```python
from llmgp import gp_run_with_pset
res = gp_run_with_pset(pset=my_pset, cost_fn=my_cost_fn, prompt="task context")
```

## Simulation-driven tasks (FunctionalSimulator)
Use this when evaluating a candidate program requires running a simulator (e.g., CRP/YCS/CTTS).

Steps:
1) `data_loader` → return a list of `(state, target)` pairs for type inference (target can be 0 if unused).
2) `feature_fns` → functions reading `state` and returning `float` features.
3) `cost_fn(pf)` → run your simulator with the candidate `pf`; return a scalar (lower is better) or a tuple/list where the first element is optimized.
4) (Optional) `extra_primitives` / `extra_terminals` → add custom ops/constants.
5) Configure and run:
```python
from llmgp import SimulatorRunner, SimulatorConfig

def data_loader():
    # Example: dummy states for type inference
    return [({"x": 1.0}, 0.0), ({"x": 2.0}, 0.0)]

def feat_x(state): return float(state["x"])

def cost_fn(pf):
    # Replace with your simulator logic; lower is better
    return sum((pf(s) - 1.0) ** 2 for s, _ in data_loader()) / 2

cfg = SimulatorConfig(prompt="My simulation task", pop_size=20, ngen=10)
sim = SimulatorRunner(
    data_loader=data_loader,
    feature_fns=[feat_x],
    cost_fn=cost_fn,
    config=cfg,
    # extra_primitives=[...], extra_terminals=[...],  # optional
)
result = sim.run()
best_pf = sim.best_pf()
print("Best rule:", result.hof[0], "score:", result.hof[0].sim_score)
```
Notes:
- `cost_fn` may return `(primary, secondary, ...)`; only the first is optimized.
- If you prefer subclassing instead of passing callables, use `SimulatorTemplate` and override `load_data`, `feature_fns`, and optionally `cost_fn`.
- To enable LLM scoring, set `LLM_HOST`, `LLM_API_KEY`, `LLM_MODEL` (otherwise LLM scores default to 0).

## Templates
- `examples/task_template.py`: fill `load_data`, feature fns, and `cost_fn`.


## LLM config (optional)
Set env vars to enable LLM scoring (fallback to 0.0 if unset):
- `LLM_HOST`
- `LLM_API_KEY`
- `LLM_MODEL`

## Web demo (FastAPI + HTML)
Run a lightweight demo server that serves `index.html` and exposes a job API for YCS/CRP.

```
python -m venv .venv
.\.venv\Scripts\activate
pip install fastapi uvicorn
uvicorn webapp:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Runs can take ~30 minutes; keep the page open to watch status.
