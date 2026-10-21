# ===--------------------------------------------------------------------------------------===#
#
# Graph coloring scoring function for CodeEvolve.
#
# The evolved program receives a graph and returns one color per vertex. This module
# turns that assignment (plus the wall-clock time it took) into a single fitness value
# that CodeEvolve maximizes, and a metrics dict the LLM can read.
#
# Fitness, one formula for every case (higher is better):
#
#     fitness = (1 - c) * (n + 1) - k - t/T
#
#   n = vertices, m = edges, c = conflicting edges, k = distinct colors used,
#   t = wall-clock seconds, T = time limit (t < T, else the run is a timeout).
#
#   Each conflict costs n + 1, which exceeds the largest possible k + t/T, so the
#   ordering is: fewer conflicts, then fewer colors, then less time.
#
#   valid (c = 0):      (n + 1 - k) - t/T                in (0, n]
#   improper (c >= 1):  (1 - c)(n + 1) - k - t/T          in (-m(n+1), -1)
#   failure:            the formula at its floor, c = m + 1, k = n + 1, t = T,
#                       i.e. -(m + 1)(n + 1) - 1, below every improper coloring.
#   Failure = exception, timeout, wrong shape or non-integer colors.
#
# ===--------------------------------------------------------------------------------------===#

from __future__ import annotations

import sys
from numbers import Integral
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------

def read_dimacs(path: str) -> tuple[int, list[tuple[int, int]]]:
    """Parse a DIMACS .col file. Returns (n, edges) with 1-indexed vertices.

    Self-loops are dropped and duplicate edges collapsed, so the conflict count
    is a count of distinct edges.
    """
    n = None
    edges: set[tuple[int, int]] = set()
    with open(path) as f:
        for line in f:
            if not line.strip() or line[0] == "c":
                continue
            parts = line.split()
            if parts[0] == "p":
                n = int(parts[2])
            elif parts[0] == "e":
                u, v = int(parts[1]), int(parts[2])
                if u != v:
                    edges.add((min(u, v), max(u, v)))
    if n is None:
        raise ValueError(f"{path}: no 'p edge N M' line")
    return n, sorted(edges)


# ---------------------------------------------------------------------------
# Assignment checking
# ---------------------------------------------------------------------------

def check_assignment(n: int, edges: Sequence[tuple[int, int]], assignment: Any):
    """Validate the shape of an assignment and count conflicts.

    Returns (colors, k, conflicts, reason). colors is None and reason is set when
    the assignment is unusable (wrong length, non-integer entries, not a sequence).
    """
    try:
        colors = list(assignment)
    except TypeError:
        return None, None, None, f"assignment is not a sequence ({type(assignment).__name__})"
    if len(colors) != n:
        return None, None, None, f"assignment has {len(colors)} entries, graph has {n} vertices"
    for i, c in enumerate(colors):
        if isinstance(c, bool) or not isinstance(c, Integral):
            return None, None, None, f"color of vertex {i + 1} is not an integer: {c!r}"
    colors = [int(c) for c in colors]
    conflicts = sum(1 for u, v in edges if colors[u - 1] == colors[v - 1])
    k = len(set(colors))
    return colors, k, conflicts, None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def fitness_formula(n: int, c: int, k: int, frac: float) -> float:
    """fitness = (1 - c)(n + 1) - k - t/T, with frac = t/T."""
    return float((1 - c) * (n + 1) - k - frac)


def failure_fitness(n: int, m: int) -> float:
    """Fitness for a run that produced no usable assignment: the formula at its
    floor, one conflict more than any coloring can have, every vertex its own
    color plus one, and the whole time budget used."""
    return fitness_formula(n, m + 1, n + 1, 1.0)


def score(
    n: int,
    edges: Sequence[tuple[int, int]],
    assignment: Any,
    elapsed: float,
    time_limit: float,
    error: str | None = None,
) -> dict:
    """Score one assignment on one graph.

    Args:
        n: number of vertices.
        edges: list of (u, v), 1-indexed.
        assignment: whatever the evolved program returned; expected to be a
            sequence of n integers, entry i being the color of vertex i+1.
        elapsed: wall-clock seconds the program took.
        time_limit: per-graph time limit T in seconds. elapsed >= T is a timeout.
        error: if the program raised, the error text; forces a failure score.

    Returns:
        dict with 'fitness' (float, higher is better) and metrics:
        valid, k, conflicts, time, time_limit, n, m, status, reason.
    """
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    m = len(edges)
    base = {"n": n, "m": m, "time": float(elapsed), "time_limit": float(time_limit)}

    def fail(status: str, reason: str) -> dict:
        return {**base, "fitness": failure_fitness(n, m), "valid": False,
                "k": None, "conflicts": None, "status": status, "reason": reason}

    if error is not None:
        return fail("error", error)
    if elapsed >= time_limit:
        return fail("timeout", f"took {elapsed:.3f}s, limit {time_limit:.3f}s")

    colors, k, conflicts, reason = check_assignment(n, edges, assignment)
    if colors is None:
        return fail("bad_output", reason)

    frac = elapsed / time_limit  # in [0, 1)
    fitness = fitness_formula(n, conflicts, k, frac)
    if conflicts == 0:
        status, reason = "valid", f"valid {k}-coloring in {elapsed:.3f}s"
    else:
        status, reason = "improper", f"{conflicts} conflicting edges, {k} colors, {elapsed:.3f}s"

    return {**base, "fitness": fitness, "valid": conflicts == 0,
            "k": k, "conflicts": conflicts, "status": status, "reason": reason}


# ---------------------------------------------------------------------------
# CLI: score a coloring file against a graph
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Score a coloring file for a DIMACS graph.")
    ap.add_argument("graph", help="DIMACS .col file")
    ap.add_argument("coloring", help="file with one integer per vertex (whitespace separated)")
    ap.add_argument("--time", type=float, default=0.0, help="elapsed seconds to attribute")
    ap.add_argument("--limit", type=float, default=60.0, help="time limit T in seconds")
    args = ap.parse_args()

    n, edges = read_dimacs(args.graph)
    with open(args.coloring) as f:
        assignment = [int(tok) for tok in f.read().split()]
    result = score(n, edges, assignment, args.time, args.limit)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["valid"] else 1)
