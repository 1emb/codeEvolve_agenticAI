# ===--------------------------------------------------------------------------------------===#
#
# Multi-instance, multi-seed graph coloring evaluator for CodeEvolve (whole-program setting).
#
# Runs the evolved program's color_graph(n, edges, time_limit) on every training graph listed
# in problem.json, once per evaluation seed, scores each run with scoring.py, and reports as
# fitness the NEGATIVE MEAN PERCENTAGE GAP to the best-known chromatic number:
#
#     gap(graph, seed) = 100 * (k - best) / best        valid coloring
#     gap(graph)       = mean over seeds
#     fitness          = -mean over graphs               higher is better, 0 = all best-known
#
# Invalid results are mapped to gaps worse than any valid coloring, ordered the same way
# scoring.py orders them (conflicts first, then failures):
#     improper (c >= 1):                     gap = 100 * (n + 1 - best) / best + 100 * c
#     failure (error, timeout, bad output):  gap = 100 * (n + 1 - best) / best + 100 * (m + 1)
#
# Several seeds per graph stop the search from selecting programs that are lucky once: seed 0
# uses the graph as given; every other seed presents an isomorphic copy (random vertex
# relabelling, shuffled edge order), so even a program with its own fixed random seed gets a
# different run. The (graph, seed) runs execute in parallel single-threaded worker processes
# (fork), each with its own SIGALRM deadline; CodeEvolve's resource monitor covers the whole
# process tree, so its eval_timeout must be >= time_limit * graphs * seeds (CPU time is summed).
# `random` and `numpy.random` are seeded with the run's seed before each call.
#
# problem.json: {"time_limit": 10.0, "seeds": [0, 1, 2], "instances": [...]}  (seeds default [0])
# Invoked by CodeEvolve as:  python evaluate.py <program.py> <results.json>
#
# ===--------------------------------------------------------------------------------------===#

import os

# Single-threaded numpy/BLAS: one core per run, CPU time ~= wall time.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import multiprocessing
import random
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from scoring import read_dimacs, score  # noqa: E402

PROBLEM_FILE = os.path.join(SCRIPT_DIR, "problem.json")
MAX_WORKERS = 9

_PROGRAM = None  # set in the parent before forking; inherited by the workers


class ProgramTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise ProgramTimeout("color_graph exceeded the time limit")


def _init_worker():
    # die with the parent even if the parent is SIGKILLed (Linux), belt and braces on top of
    # CodeEvolve's own process-tree kill
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        pass


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


def run_one(job) -> dict:
    """One (graph, seed) run of the program loaded in _PROGRAM. Never raises."""
    graph_path, time_limit, seed = job
    n, edges = read_dimacs(graph_path)
    edges0 = [(u - 1, v - 1) for u, v in edges]
    if seed != 0:
        prm = random.Random(seed)
        perm = list(range(n))
        prm.shuffle(perm)
        edges0 = [(perm[u], perm[v]) for u, v in edges0]
        prm.shuffle(edges0)
        edges = sorted((min(u, v) + 1, max(u, v) + 1) for u, v in edges0)   # scored on the same copy
    assignment, error, elapsed, t0 = None, None, 0.0, None
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, time_limit)
        t0 = time.perf_counter()
        assignment = _PROGRAM.color_graph(n, edges0, time_limit)
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
    res = score(n, edges, assignment, elapsed, time_limit, error=error)
    res["seed"] = seed
    return res


def gap_pct(res: dict, best: int) -> float:
    n, m = res["n"], res["m"]
    worst_valid = 100.0 * (n + 1 - best) / best
    if res["status"] == "valid":
        return 100.0 * (res["k"] - best) / best
    if res["status"] == "improper":
        return worst_valid + 100.0 * res["conflicts"]
    return worst_valid + 100.0 * (m + 1)


def failed_run(graph_path, time_limit, seed, error):
    n, edges = read_dimacs(graph_path)
    res = score(n, edges, None, 0.0, time_limit, error=error)
    res["seed"] = seed
    return res


def run_all(jobs):
    """Run every (graph_path, time_limit, seed) job; parallel when there is more than one."""
    global _PROGRAM
    if len(jobs) == 1:
        return [run_one(jobs[0])]
    results = [None] * len(jobs)
    try:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs)), mp_context=ctx,
                                 initializer=_init_worker) as ex:
            futs = {ex.submit(run_one, job): i for i, job in enumerate(jobs)}
            for fut, i in futs.items():
                try:
                    results[i] = fut.result()
                except BrokenProcessPool:
                    raise
                except BaseException as err:  # noqa: BLE001
                    results[i] = failed_run(*jobs[i], error=f"{type(err).__name__}: {err}")
    except BrokenProcessPool as err:
        for i, job in enumerate(jobs):
            if results[i] is None:
                results[i] = failed_run(*job, error=f"worker process died (crash or kill): {err}")
    return results


def evaluate(program_path: str, results_path: str) -> None:
    global _PROGRAM
    problem = load_problem()
    time_limit = float(problem["time_limit"])
    seeds = [int(s) for s in problem.get("seeds", [0])]
    instances = problem["instances"]

    load_error = None
    try:
        _PROGRAM = load_program(program_path)
    except BaseException as err:  # noqa: BLE001
        load_error = f"{type(err).__name__}: {err}\n{traceback.format_exc(limit=5)[-1500:]}"
        _PROGRAM = None

    t_all = time.perf_counter()
    jobs = [(os.path.join(SCRIPT_DIR, inst["file"]), time_limit, s) for inst in instances for s in seeds]
    if _PROGRAM is None:
        runs = [failed_run(*job, error=load_error) for job in jobs]
    else:
        runs = run_all(jobs)
    total_time = time.perf_counter() - t_all

    per_instance = []
    for idx, inst in enumerate(instances):
        rs = runs[idx * len(seeds):(idx + 1) * len(seeds)]
        gaps = [gap_pct(r, inst["best_known"]) for r in rs]
        statuses = [r["status"] for r in rs]
        per_instance.append({
            "instance": inst["name"], "best_known": inst["best_known"],
            "k": [r.get("k") for r in rs],
            "gap_pct": sum(gaps) / len(gaps),
            "time": sum(r["time"] for r in rs) / len(rs),
            "status": "valid" if all(s == "valid" for s in statuses) else next(s for s in statuses if s != "valid"),
            "reason": "; ".join(f"seed {r['seed']}: {r['reason']}" for r in rs),
        })

    mean_gap = sum(p["gap_pct"] for p in per_instance) / len(per_instance)
    result = {
        "fitness": -mean_gap,
        "mean_gap_pct": mean_gap,
        "all_valid": all(p["status"] == "valid" for p in per_instance),
        "instances": [p["instance"] for p in per_instance],
        "best_known": [p["best_known"] for p in per_instance],
        "seeds": seeds,
        "k": [p["k"] for p in per_instance],
        "k_mean": [round(sum(k for k in p["k"] if k is not None) / len(p["k"]), 2) if all(k is not None for k in p["k"]) else None for p in per_instance],
        "gap_pct": [round(p["gap_pct"], 3) for p in per_instance],
        "time": [round(p["time"], 3) for p in per_instance],
        "status": [p["status"] for p in per_instance],
        "reason": [p["reason"] for p in per_instance],
        "time_limit_per_graph": time_limit,
        "eval_time": total_time,
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])
