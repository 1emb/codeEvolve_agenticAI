# ===--------------------------------------------------------------------------------------===#
#
# Fixed-k graph coloring evaluator for CodeEvolve ("hook" problem).
#
# For every training graph in problem.json the evolved rules (select_move, tenure) are run
# inside the fixed TabuCol framework at a fixed target number of colours for a fixed number
# of iterations. Per graph:
#     solved:    score = iterations_used / budget        in (0, 1], faster is better
#     unsolved:  score = conflicting edges left           >= 1, fewer is better
#     failure:   score = m + 1                           worse than any search result
#     fitness = -mean(score)                             0 = every target solved instantly
#
# Deterministic: the same rules give the same score on any machine (iteration budget, seeded
# rng, single-threaded numpy). A per-graph wall-clock cap stops rules whose iterations are
# absurdly expensive. Every failure mode is caught so CodeEvolve always gets a results file.
#
# Invoked by CodeEvolve as:  python evaluate.py <program.py> <results.json>
#
# ===--------------------------------------------------------------------------------------===#

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import random
import signal
import sys
import time
import traceback

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import tabucol_framework as fw  # noqa: E402
from scoring import read_dimacs  # noqa: E402

PROBLEM_FILE = os.path.join(SCRIPT_DIR, "problem.json")
SEED = 0


class WallCapExceeded(Exception):
    pass


def _on_alarm(signum, frame):
    raise WallCapExceeded("per-graph wall-clock cap exceeded")


def load_program(program_path):
    program_dir = os.path.dirname(os.path.abspath(program_path))
    module_name = os.path.splitext(os.path.basename(program_path))[0]
    sys.path.insert(0, program_dir)
    try:
        program = __import__(module_name)
    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)
    for name in ("select_move", "tenure"):
        if not callable(getattr(program, name, None)):
            raise AttributeError(f"program does not define {name}()")
    return program


def count_conflicts(edges0, col):
    return int(sum(1 for u, v in edges0 if col[u] == col[v]))


def run_graph(program, inst, budget, wall_cap):
    graph_path = os.path.join(SCRIPT_DIR, inst["file"])
    n, edges = read_dimacs(graph_path)
    edges0 = [(u - 1, v - 1) for u, v in edges]
    m = len(edges0)
    k = int(inst["target_k"])
    res = {"instance": inst["name"], "n": n, "m": m, "target_k": k, "best_known": inst["best_known"],
           "best_conflicts": None, "solved_at": None, "iters": 0, "wall": 0.0, "status": "error", "reason": ""}

    random.seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)
    adj_idx = fw.build_adjacency(n, edges0)
    col = fw.start_coloring(fw.dsatur(n, adj_idx), k, rng)

    t0 = time.perf_counter()
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, wall_cap)
        best_col, best_conf, solved_at, iters = fw.fixed_k_search(
            n, adj_idx, k, col, budget, rng, program.select_move, program.tenure)
        signal.setitimer(signal.ITIMER_REAL, 0)
        res["wall"] = time.perf_counter() - t0
        # independent verification of what the framework reports
        best_col = np.asarray(best_col)
        if best_col.shape != (n,) or int(best_col.min()) < 0 or int(best_col.max()) >= k:
            raise ValueError("best colouring is not a k-colouring")
        verified = count_conflicts(edges0, best_col)
        if verified != best_conf:
            raise ValueError(f"reported {best_conf} conflicts but colouring has {verified}")
        res.update(best_conflicts=int(best_conf), solved_at=solved_at, iters=int(iters))
        if solved_at is not None:
            res["score"] = solved_at / budget
            res["status"] = "solved"
            res["reason"] = f"legal {k}-colouring after {solved_at:,} of {budget:,} iterations ({res['wall']:.1f}s)"
        else:
            res["score"] = float(best_conf)
            res["status"] = "unsolved"
            res["reason"] = f"{best_conf} conflicting edges left at k={k} after {iters:,} iterations ({res['wall']:.1f}s)"
    except BaseException as err:  # noqa: BLE001
        signal.setitimer(signal.ITIMER_REAL, 0)
        res["wall"] = time.perf_counter() - t0
        res["score"] = float(m + 1)
        if isinstance(err, WallCapExceeded):
            res["status"] = "timeout"
            res["reason"] = f"exceeded the {wall_cap:g}s wall-clock cap for one graph (iterations too expensive)"
        else:
            res["status"] = "error"
            tb = traceback.format_exc(limit=4)
            res["reason"] = f"{type(err).__name__}: {err}\n{tb[-800:]}"
    return res


def evaluate(program_path, results_path):
    with open(PROBLEM_FILE) as f:
        problem = json.load(f)
    budget = int(problem["budget"])
    wall_cap = float(problem["wall_cap"])
    instances = problem["instances"]

    t_all = time.perf_counter()
    try:
        program = load_program(program_path)
        per = [run_graph(program, inst, budget, wall_cap) for inst in instances]
    except BaseException as err:  # noqa: BLE001
        msg = f"{type(err).__name__}: {err}\n{traceback.format_exc(limit=4)[-800:]}"
        per = []
        for inst in instances:
            n, edges = read_dimacs(os.path.join(SCRIPT_DIR, inst["file"]))
            per.append({"instance": inst["name"], "n": n, "m": len(edges), "target_k": inst["target_k"],
                        "best_known": inst["best_known"], "best_conflicts": None, "solved_at": None,
                        "iters": 0, "wall": 0.0, "score": float(len(edges) + 1), "status": "error", "reason": msg})
    eval_time = time.perf_counter() - t_all

    scores = [r["score"] for r in per]
    result = {
        "fitness": -sum(scores) / len(scores),
        "score": [round(s, 4) for s in scores],
        "instances": [r["instance"] for r in per],
        "target_k": [r["target_k"] for r in per],
        "best_known": [r["best_known"] for r in per],
        "best_conflicts": [r["best_conflicts"] for r in per],
        "solved_at": [r["solved_at"] for r in per],
        "iters": [r["iters"] for r in per],
        "wall": [round(r["wall"], 2) for r in per],
        "status": [r["status"] for r in per],
        "reason": [r["reason"] for r in per],
        "budget": budget,
        "eval_time": eval_time,
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])
