# ===--------------------------------------------------------------------------------------===#
#
# Graph coloring evaluator for CodeEvolve.
#
# Runs the evolved program's color_graph(n, edges, time_limit) on the graph in this
# directory, scores the returned assignment with scoring.py, and writes the metrics JSON.
# Every failure mode (import error, exception, timeout, bad output) is caught here and
# written as a failure fitness, so CodeEvolve always receives a results file.
#
# Invoked by CodeEvolve as:  python evaluate.py <program.py> <results.json>
#
# ===--------------------------------------------------------------------------------------===#

import json
import os
import signal
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from scoring import read_dimacs, score  # noqa: E402

GRAPH_FILE = os.path.join(SCRIPT_DIR, "graph.col")
PROBLEM_FILE = os.path.join(SCRIPT_DIR, "problem.json")


class ProgramTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise ProgramTimeout("color_graph exceeded the time limit")


def load_problem() -> dict:
    with open(PROBLEM_FILE) as f:
        return json.load(f)


def evaluate(program_path: str, results_path: str) -> None:
    problem = load_problem()
    time_limit = float(problem["time_limit"])
    n, edges = read_dimacs(GRAPH_FILE)
    edges0 = [(u - 1, v - 1) for u, v in edges]  # program sees 0-indexed vertices

    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]

    assignment, error = None, None
    elapsed = 0.0
    t0 = None
    try:
        sys.path.insert(0, program_dir)
        try:
            program = __import__(module_name)
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)
        if not hasattr(program, "color_graph"):
            raise AttributeError("program does not define color_graph(n, edges, time_limit)")

        signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, time_limit)
        t0 = time.perf_counter()
        assignment = program.color_graph(n, edges0, time_limit)
        elapsed = time.perf_counter() - t0
    except BaseException as err:  # noqa: BLE001  (SystemExit/KeyboardInterrupt from the program too)
        elapsed = time.perf_counter() - t0 if t0 is not None else 0.0
        if isinstance(err, ProgramTimeout):
            elapsed = max(elapsed, time_limit)   # scoring reports this as status 'timeout'
        else:
            tb = traceback.format_exc(limit=5)
            error = f"{type(err).__name__}: {err}\n{tb[-1500:]}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

    result = score(n, edges, assignment, elapsed, time_limit, error=error)
    result["eval_time"] = result["time"]          # MAP-Elites feature name used by the configs
    result["instance"] = problem.get("instance")
    result["best_known_k"] = problem.get("best_known")

    with open(results_path, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])
