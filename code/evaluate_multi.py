# ===--------------------------------------------------------------------------------------===#
#
# Multi-instance graph coloring evaluator for CodeEvolve (EoH-style heuristic setting).
#
# Runs the evolved program's color_graph(n, edges, time_limit) on every training graph
# listed in problem.json, scores each with scoring.py, and reports as fitness the
# NEGATIVE MEAN PERCENTAGE GAP to the best-known chromatic number, the same way the
# CodeEvolve TSP-GLS benchmark uses the negative mean gap to the optimal tour:
#
#     gap_i   = 100 * (k_i - best_i) / best_i            valid coloring of graph i
#     fitness = -mean_i(gap_i)                           higher is better, 0 = all best-known
#
# Invalid results are mapped to gaps that are worse than any valid coloring and ordered
# the same way scoring.py orders them (conflicts first, then failures):
#     improper (c >= 1):   gap = 100 * (n + 1 - best) / best + 100 * c
#     failure (error, timeout, bad output):  gap = 100 * (n + 1 - best) / best + 100 * (m + 1)
#
# Every failure mode is caught so CodeEvolve always receives a results file.
# `random` and numpy are seeded before each graph so an evaluation is reproducible.
#
# Invoked by CodeEvolve as:  python evaluate.py <program.py> <results.json>
#
# ===--------------------------------------------------------------------------------------===#

import os

# Keep numpy/BLAS single-threaded so CPU time ~= wall time: CodeEvolve enforces its
# eval_timeout on CPU time of the process, and 10 evaluations run in parallel.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import random
import signal
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from scoring import read_dimacs, score  # noqa: E402

PROBLEM_FILE = os.path.join(SCRIPT_DIR, "problem.json")
SEED = 0


class ProgramTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise ProgramTimeout("color_graph exceeded the time limit")


def load_problem() -> dict:
    with open(PROBLEM_FILE) as f:
        return json.load(f)


def load_program(program_path: str):
    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]
    sys.path.insert(0, program_dir)
    try:
        program = __import__(module_name)
    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)
    if not hasattr(program, "color_graph"):
        raise AttributeError("program does not define color_graph(n, edges, time_limit)")
    return program


def run_one(program, graph_path: str, time_limit: float) -> dict:
    n, edges = read_dimacs(graph_path)
    edges0 = [(u - 1, v - 1) for u, v in edges]
    assignment, error, elapsed, t0 = None, None, 0.0, None
    random.seed(SEED)
    try:
        import numpy as np
        np.random.seed(SEED)
    except Exception:  # noqa: BLE001
        pass
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, time_limit)
        t0 = time.perf_counter()
        assignment = program.color_graph(n, edges0, time_limit)
        elapsed = time.perf_counter() - t0
    except BaseException as err:  # noqa: BLE001
        elapsed = time.perf_counter() - t0 if t0 is not None else 0.0
        if isinstance(err, ProgramTimeout):
            elapsed = max(elapsed, time_limit)
        else:
            tb = traceback.format_exc(limit=5)
            error = f"{type(err).__name__}: {err}\n{tb[-1500:]}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return score(n, edges, assignment, elapsed, time_limit, error=error)


def gap_pct(res: dict, best: int) -> float:
    n, m = res["n"], res["m"]
    worst_valid = 100.0 * (n + 1 - best) / best
    if res["status"] == "valid":
        return 100.0 * (res["k"] - best) / best
    if res["status"] == "improper":
        return worst_valid + 100.0 * res["conflicts"]
    return worst_valid + 100.0 * (m + 1)


def evaluate(program_path: str, results_path: str) -> None:
    problem = load_problem()
    time_limit = float(problem["time_limit"])
    instances = problem["instances"]

    per_instance, load_error = [], None
    try:
        program = load_program(program_path)
    except BaseException as err:  # noqa: BLE001
        load_error = f"{type(err).__name__}: {err}\n{traceback.format_exc(limit=5)[-1500:]}"
        program = None

    t_all = time.perf_counter()
    for inst in instances:
        graph_path = os.path.join(SCRIPT_DIR, inst["file"])
        if program is None:
            n, edges = read_dimacs(graph_path)
            res = score(n, edges, None, 0.0, time_limit, error=load_error)
        else:
            res = run_one(program, graph_path, time_limit)
        res["instance"] = inst["name"]
        res["best_known_k"] = inst["best_known"]
        res["gap_pct"] = gap_pct(res, inst["best_known"])
        per_instance.append(res)
    total_time = time.perf_counter() - t_all

    gaps = [r["gap_pct"] for r in per_instance]
    mean_gap = sum(gaps) / len(gaps)
    result = {
        "fitness": -mean_gap,
        "mean_gap_pct": mean_gap,
        "all_valid": all(r["status"] == "valid" for r in per_instance),
        "instances": [r["instance"] for r in per_instance],
        "best_known": [r["best_known_k"] for r in per_instance],
        "k": [r.get("k") for r in per_instance],
        "gap_pct": [round(g, 3) for g in gaps],
        "time": [round(r["time"], 3) for r in per_instance],
        "status": [r["status"] for r in per_instance],
        "reason": [r["reason"] for r in per_instance],
        "time_limit_per_graph": time_limit,
        "eval_time": total_time,
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])
