#!/usr/bin/env python
"""Build a self-contained CodeEvolve problem directory for one DIMACS graph.

    python make_problem.py DSJC125.5 [--time-limit 30] [--best-known 17] [--epochs 100]
                           [--islands 5] [--model Qwen/Qwen3-Coder-30B-A3B-Instruct]

Creates problems/graph_coloring/<instance>/
    input/evaluate.py, input/scoring.py, input/graph.col, input/problem.json,
    input/src/init_program.py, config_qwen.yaml, config_mock.yaml

The graph is taken from $CODEEVOLVE_DATA/dimacs/<instance>.col (default
~/codeEvolve/data/dimacs) and downloaded from the COLOR02 site if missing.
"""
import argparse
import json
import os
import shutil
import sys
import urllib.request

import yaml

COMMON = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = os.path.dirname(COMMON)
DATA = os.environ.get("CODEEVOLVE_DATA", os.path.expanduser("~/codeEvolve/data"))
COLOR02 = "https://mat.tepper.cmu.edu/COLOR02/INSTANCES/"

# best-known / optimal colors (Goudet & Hao 2021, Titiloye QA-COL certificates, COLOR02 table)
BEST_KNOWN = {
    "myciel3": 4, "myciel4": 5, "myciel5": 6, "myciel6": 7, "myciel7": 8,
    "queen5_5": 5, "queen6_6": 7, "queen7_7": 7, "queen8_8": 9, "queen9_9": 10, "queen10_10": 11,
    "DSJC125.1": 5, "DSJC125.5": 17, "DSJC125.9": 44,
    "DSJC250.1": 8, "DSJC250.5": 28, "DSJC250.9": 72,
    "DSJC500.1": 12, "DSJC500.5": 47, "DSJC500.9": 126,
    "DSJC1000.1": 20, "DSJC1000.5": 82, "DSJC1000.9": 222,
    "DSJR500.1c": 85, "DSJR500.5": 122,
    "le450_15c": 15, "le450_15d": 15, "le450_25c": 25, "le450_25d": 25,
    "flat300_28_0": 28, "flat1000_50_0": 50, "flat1000_60_0": 60, "flat1000_76_0": 81,
    "latin_square_10": 97, "C2000.5": 145, "C4000.5": 259,
}


def sys_msg(instance, n, m, best_known, time_limit):
    bk = f"The best coloring known in the literature uses {best_known} colors." if best_known else ""
    return f"""# PROMPT-BLOCK-START
SETTING:
You are an expert in combinatorial optimization and graph algorithms. You iteratively improve a Python program that colors the vertices of an undirected graph so that no edge joins two vertices of the same color, using as few colors as possible, within a fixed time budget.

PROBLEM INSTANCE:
Graph {instance}: n = {n} vertices, m = {m} edges. {bk}

PROGRAM CONTRACT:
- The program defines color_graph(n, edges, time_limit). n is the number of vertices (labelled 0..n-1), edges is a list of (u, v) tuples, time_limit is the wall-clock budget in seconds.
- It must return a list of n integers, entry i being the color of vertex i. Any integer labels are allowed, only equality matters.
- It must return before time_limit seconds elapse, otherwise the run is a failure. Use the budget: a local search that keeps improving until the deadline is allowed and encouraged.
- Only the Python standard library and numpy are available. No file, network or subprocess access.

PERFORMANCE METRICS (reported for every evaluation):
1. **fitness**: (1 - conflicts) * (n + 1) - k - time/time_limit. Higher is better. A valid coloring scores (n + 1 - k) - time/time_limit, so using one fewer color always beats any speed-up, and among equal k the faster program wins. Any conflicting edge makes the fitness negative; a crash, timeout or malformed output is worse than any coloring.
2. **k**: number of distinct colors used (lower is better).
3. **conflicts**: number of edges whose endpoints share a color (must be 0).
4. **time**: seconds color_graph took (limit {time_limit}).
5. **status** / **reason**: valid, improper, timeout, error or bad_output, with details.
# PROMPT-BLOCK-END
"""


def base_config(instance, n, m, best_known, time_limit, epochs, islands):
    return {
        "SYS_MSG": sys_msg(instance, n, m, best_known, time_limit),
        "CODEBASE_PATH": "src/",
        "INIT_FILE_DATA": {"filename": "init_program.py", "language": "python"},
        "EVAL_FILE_NAME": "evaluate.py",
        "BUDGET_CONFIG": {
            "eval_timeout": int(time_limit) + 30,     # program limit is enforced inside evaluate.py
            "max_mem_bytes": 2_000_000_000,
            "resource_check_interval_s": 0.1,
        },
        "EVOLVE_CONFIG": {
            "fitness_key": "fitness",
            "num_epochs": epochs, "ckpt": 10, "max_size": None, "init_pop": 10,
            "exploration_rate": 0.2,
            "selection": {"policy": "roulette", "kwargs": {"roulette_by_rank": True}},
            "early_stopping_rounds": 40,
            "num_islands": islands,
            "migration": {"topology": "ring", "interval": 25, "rate": 0.1},
            "meta_prompting": True, "use_map_elites": True,
            "num_inspirations": 2,
            "max_chat_depth": 5,
            "markers": {"mp_start_marker": "# PROMPT-BLOCK-START", "mp_end_marker": "# PROMPT-BLOCK-END",
                        "evolve_start_marker": "# EVOLVE-BLOCK-START", "evolve_end_marker": "# EVOLVE-BLOCK-END"},
            "exploration_scheduler": {"type": "PlateauScheduler",
                                      "kwargs": {"min_value": 0.2, "max_value": 0.5, "plateau_threshold": 5,
                                                 "increase_factor": 1.05, "decrease_factor": 0.95}},
        },
        "MAP_ELITES": {"elite_map_type": "grid",
                       "features": [{"name": "eval_time", "min_val": 0, "max_val": time_limit, "num_bins": 20}]},
    }


def lm(model, temp):
    return {"model_name": model, "temp": temp, "top_p": 0.8, "max_tok": 8192, "retries": 3, "weight": 1, "verify_ssl": False}


class BlockStr(str):
    pass


def _block_repr(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(BlockStr, _block_repr)


def write_yaml(path, cfg):
    cfg = dict(cfg)
    cfg["SYS_MSG"] = BlockStr(cfg["SYS_MSG"])
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False, width=120)


def ensure_graph(instance):
    os.makedirs(os.path.join(DATA, "dimacs"), exist_ok=True)
    path = os.path.join(DATA, "dimacs", f"{instance}.col")
    if not os.path.exists(path):
        url = f"{COLOR02}{instance}.col"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--best-known", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--islands", type=int, default=5)
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--force", action="store_true", help="overwrite an existing problem dir")
    a = ap.parse_args()

    sys.path.insert(0, COMMON)
    from scoring import read_dimacs

    graph = ensure_graph(a.instance)
    n, edges = read_dimacs(graph)
    m = len(edges)
    best = a.best_known if a.best_known is not None else BEST_KNOWN.get(a.instance)

    prob = os.path.join(PROBLEMS, a.instance)
    inp = os.path.join(prob, "input")
    if os.path.exists(prob) and not a.force:
        sys.exit(f"{prob} exists; use --force to overwrite")
    os.makedirs(os.path.join(inp, "src"), exist_ok=True)
    shutil.copy(os.path.join(COMMON, "evaluate.py"), inp)
    shutil.copy(os.path.join(COMMON, "scoring.py"), inp)
    shutil.copy(os.path.join(COMMON, "init_program.py"), os.path.join(inp, "src", "init_program.py"))
    shutil.copy(graph, os.path.join(inp, "graph.col"))
    with open(os.path.join(inp, "problem.json"), "w") as f:
        json.dump({"instance": a.instance, "n": n, "m": m, "best_known": best,
                   "time_limit": a.time_limit}, f, indent=2)

    cfg = base_config(a.instance, n, m, best, a.time_limit, a.epochs, a.islands)
    qwen = dict(cfg, EXPLORATION_ENSEMBLE=[lm(a.model, 0.9)], EXPLOITATION_ENSEMBLE=[lm(a.model, 0.7)],
                SAMPLER_AUX_LM=lm(a.model, 0.7))
    mock = dict(cfg, EXPLORATION_ENSEMBLE=[{"model_name": "MOCK"}], EXPLOITATION_ENSEMBLE=[{"model_name": "MOCK"}],
                SAMPLER_AUX_LM={"model_name": "MOCK"})
    write_yaml(os.path.join(prob, "config_qwen.yaml"), qwen)
    write_yaml(os.path.join(prob, "config_mock.yaml"), mock)
    print(f"built {prob}: n={n} m={m} best_known={best} time_limit={a.time_limit}s "
          f"epochs={a.epochs} islands={a.islands}")


if __name__ == "__main__":
    main()
