#!/usr/bin/env python
# ===--------------------------------------------------------------------------------------===#
#
# Build a multi-instance graph coloring problem for CodeEvolve in the EoH-style
# "heuristic discovery" setting used by the CodeEvolve paper (Sec. 5.3, TSP-GLS config):
#   - a small training set of graphs (3 by default), each with its own time limit
#   - fitness = negative mean percentage gap to the best-known chromatic number
#   - 10 islands, 152 epochs, init_pop 10, no early stopping, exploration 0.3 with the
#     PlateauScheduler, chat depth 3, 2 inspirations, ring migration (interval 30, rate 0.1),
#     CVT MAP-Elites on (fitness, eval_time), Qwen3-Coder-30B at temp 0.7 / top_p 0.8.
#
#   python make_problem_multi.py <name> --instances DSJC250.5,le450_25c,DSJC500.5 [--time-limit 10]
#       [--init init_tabucol.py] [--epochs 152] [--islands 10] [--force]
#
# Produces problems/graph_coloring/<name>/{input/, config_qwen.yaml, config_mock.yaml}
# with input/ = {evaluate.py, scoring.py, problem.json, graphs/*.col, src/init_program.py}.
#
# ===--------------------------------------------------------------------------------------===#

import argparse
import json
import os
import shutil
import sys

import yaml

COMMON = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = os.path.dirname(COMMON)
DATA = os.path.join(os.path.expanduser("~"), "codeEvolve", "data")
sys.path.insert(0, COMMON)
from make_problem import BEST_KNOWN, BlockStr, ensure_graph, lm, write_yaml  # noqa: E402,F401
from scoring import read_dimacs  # noqa: E402


def sys_msg(instances, time_limit, eval_timeout, init_name):
    rows = "\n".join(
        f"  * {i['name']}: n = {i['n']}, m = {i['m']}, best known k = {i['best_known']}"
        for i in instances
    )
    return f"""# PROMPT-BLOCK-START
You are an expert in combinatorial optimization, metaheuristics, and graph coloring.
Your mission is to design a novel and high-performing graph coloring algorithm: given an undirected graph, assign a color to every vertex so that no edge joins two vertices of the same color, using as few colors as possible within a fixed time budget.

PROBLEM CONTEXT:
- **Task**: Improve the program that defines `color_graph(n, edges, time_limit) -> list[int]`.
- **Inputs**:
  * `n` (int): number of vertices, labelled 0..n-1.
  * `edges` (list of (u, v) tuples): undirected edges, 0-indexed.
  * `time_limit` (float): wall-clock budget in seconds for this call. The function MUST return before it elapses; returning late counts as a failure.
- **Output**: a list of n integers, entry i is the color of vertex i. Any integer labels are allowed, only equality matters. Any edge whose endpoints share a color makes the result invalid.
- **Graph setup**:
  * The program is evaluated on {len(instances)} DIMACS benchmark graphs (the training set), one call per graph, {time_limit:g} seconds each:
{rows}
  * Graphs of this family (random G(n, 0.5) and Leighton graphs) are hard: greedy methods such as DSATUR use far more colors than the best known; the best known colorings come from tabu search / hybrid evolutionary algorithms with a fixed number of colors k that is decreased whenever a legal k-coloring is found.

EVALUATION:
- **Fitness** (maximize): negative of the mean percentage gap (%) to the best-known number of colors over the training graphs, i.e. -mean(100 * (k - best_known) / best_known). A fitness of 0.0 means the best-known coloring was matched on every graph. Higher (closer to 0) is better.
- **mean_gap_pct**, **k**, **gap_pct**, **time**, **status**, **reason**: per-graph details reported for every evaluation. An invalid coloring, crash, timeout or malformed output on any graph gets a gap far worse than any valid coloring.
- **eval_time**: total execution time in seconds across all graphs.

COMPUTATIONAL RESOURCES & IMPLEMENTATION GUIDELINES:
**Core packages**: numpy and the Python standard library only. No file, network or subprocess access, no multiprocessing.
- Use the whole time budget: an algorithm that keeps improving until shortly before the deadline is expected. Check the clock regularly; leave a safety margin.
- Prefer vectorized NumPy operations for the inner loop; pure-Python loops over all vertices and colors per iteration are slow.
- `random` and `numpy.random` are seeded before each call, so the program is deterministic for reproducibility.
- The current program is {init_name}; its structure (greedy start, then fixed-k local search with k decreasing) is a known-good framework. Both small refinements (move selection, tabu tenure, restarts, when to give up on a k) and new ideas (different neighbourhoods, perturbation, crossover between colorings, smarter class removal, better use of the clock) are welcome.

COMPUTATIONAL BUDGET:
- **Time limit**: {time_limit:g} seconds per graph, {eval_timeout} seconds maximum total execution time.
- **Memory limit**: 2 GB.

KNOWN BASELINES:
- DSATUR alone: ~16-32% above the best known on these graphs.
- DSATUR + TabuCol with decreasing k (the starting program): ~5% above the best known.
- The best known values were obtained by much longer runs of tabu search and memetic algorithms.
# PROMPT-BLOCK-END
"""


def base_config(instances, time_limit, eval_timeout, epochs, islands, init_name):
    return {
        "SYS_MSG": sys_msg(instances, time_limit, eval_timeout, init_name),
        "CODEBASE_PATH": "src/",
        "INIT_FILE_DATA": {"filename": "init_program.py", "language": "python"},
        "EVAL_FILE_NAME": "evaluate.py",
        "BUDGET_CONFIG": {
            "eval_timeout": eval_timeout,
            "max_mem_bytes": 2_000_000_000,
            "resource_check_interval_s": 0.1,
        },
        "EVOLVE_CONFIG": {
            "fitness_key": "fitness",
            "num_epochs": epochs, "ckpt": 10, "max_size": None, "init_pop": 10,
            "exploration_rate": 0.3,
            "selection": {"policy": "roulette", "kwargs": {"roulette_by_rank": True}},
            "early_stopping_rounds": None,
            "num_islands": islands,
            "migration": {"topology": "ring", "interval": 30, "rate": 0.1},
            "meta_prompting": True, "use_map_elites": True,
            "num_inspirations": 2,
            "max_chat_depth": 3,
            "markers": {"mp_start_marker": "# PROMPT-BLOCK-START", "mp_end_marker": "# PROMPT-BLOCK-END",
                        "evolve_start_marker": "# EVOLVE-BLOCK-START", "evolve_end_marker": "# EVOLVE-BLOCK-END"},
            "exploration_scheduler": {"type": "PlateauScheduler",
                                      "kwargs": {"min_value": 0.2, "max_value": 0.5, "plateau_threshold": 5,
                                                 "increase_factor": 1.05, "decrease_factor": 0.95}},
        },
        "MAP_ELITES": {
            "elite_map_type": "cvt",
            "elite_map_kwargs": {"num_centroids": 30, "num_init_samples": 1000, "max_iter": 300, "tolerance": 1e-4},
            "features": [
                {"name": "fitness", "min_val": -40.0, "max_val": 0.0},
                {"name": "eval_time", "min_val": 0, "max_val": eval_timeout},
            ],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--instances", required=True, help="comma-separated DIMACS instance names")
    ap.add_argument("--time-limit", type=float, default=10.0, help="seconds per graph")
    ap.add_argument("--init", default="init_tabucol.py", help="initial program file in common/")
    ap.add_argument("--epochs", type=int, default=152)
    ap.add_argument("--islands", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    names = [s.strip() for s in a.instances.split(",") if s.strip()]
    prob = os.path.join(PROBLEMS, a.name)
    inp = os.path.join(prob, "input")
    if os.path.exists(prob) and not a.force:
        sys.exit(f"{prob} exists; use --force to overwrite")
    os.makedirs(os.path.join(inp, "src"), exist_ok=True)
    os.makedirs(os.path.join(inp, "graphs"), exist_ok=True)

    instances = []
    for name in names:
        path = ensure_graph(name)
        n, edges = read_dimacs(path)
        if name not in BEST_KNOWN:
            sys.exit(f"no best-known k for {name}; add it to BEST_KNOWN in make_problem.py")
        shutil.copy(path, os.path.join(inp, "graphs", f"{name}.col"))
        instances.append({"name": name, "file": f"graphs/{name}.col", "n": n, "m": len(edges),
                          "best_known": BEST_KNOWN[name]})

    eval_timeout = int(a.time_limit * len(instances)) + 30
    shutil.copy(os.path.join(COMMON, "evaluate_multi.py"), os.path.join(inp, "evaluate.py"))
    shutil.copy(os.path.join(COMMON, "scoring.py"), inp)
    shutil.copy(os.path.join(COMMON, a.init), os.path.join(inp, "src", "init_program.py"))
    with open(os.path.join(inp, "problem.json"), "w") as f:
        json.dump({"name": a.name, "time_limit": a.time_limit, "instances": instances}, f, indent=2)

    init_name = "DSATUR followed by TabuCol with decreasing k" if "tabucol" in a.init else "DSATUR"
    cfg = base_config(instances, a.time_limit, eval_timeout, a.epochs, a.islands, init_name)
    # paper setting: temp 0.7 / top_p 0.8 for exploration, exploitation and the meta-prompt LM
    qwen = dict(cfg, EXPLORATION_ENSEMBLE=[lm(a.model, 0.7)], EXPLOITATION_ENSEMBLE=[lm(a.model, 0.7)],
                SAMPLER_AUX_LM=lm(a.model, 0.7))
    mock = dict(cfg, EXPLORATION_ENSEMBLE=[{"model_name": "MOCK"}], EXPLOITATION_ENSEMBLE=[{"model_name": "MOCK"}],
                SAMPLER_AUX_LM={"model_name": "MOCK"})
    write_yaml(os.path.join(prob, "config_qwen.yaml"), qwen)
    write_yaml(os.path.join(prob, "config_mock.yaml"), mock)
    print(f"built {prob}: {len(instances)} graphs {names}, {a.time_limit:g}s each, eval_timeout={eval_timeout}s, "
          f"epochs={a.epochs} islands={a.islands} init={a.init}")


if __name__ == "__main__":
    main()
