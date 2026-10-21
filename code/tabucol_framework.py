# ===--------------------------------------------------------------------------------------===#
#
# Fixed TabuCol framework for the graph coloring "hook" problem. NOT evolved.
#
# The evolved program supplies two decision rules:
#     select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng) -> (v, c)
#     tenure(n, k, num_conflicting, it, rng) -> int
# This module owns everything else: adjacency, the DSATUR start, the fixed-k tabu loop
# (which counts iterations), and the decreasing-k wrapper used for held-out evaluation.
#
# ===--------------------------------------------------------------------------------------===#

import random
import time

import numpy as np


def build_adjacency(n, edges):
    """edges: (u, v) pairs, 0-indexed. Returns a list of n int64 arrays of neighbours."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return [np.fromiter(a, dtype=np.int64, count=len(a)) for a in adj]


def dsatur(n, adj_idx):
    """Greedy DSATUR: colour the uncoloured vertex with the most distinct neighbour colours
    (ties by degree) with the smallest colour its neighbours lack. Returns list of n ints."""
    color = [-1] * n
    seen = [set() for _ in range(n)]
    degree = [len(a) for a in adj_idx]
    uncolored = set(range(n))
    while uncolored:
        v = max(uncolored, key=lambda x: (len(seen[x]), degree[x]))
        c = 0
        while c in seen[v]:
            c += 1
        color[v] = c
        uncolored.remove(v)
        for w in adj_idx[v]:
            seen[int(w)].add(c)
    return color


def start_coloring(base, k, rng):
    """k-colouring (numpy int64 array) built from a legal colouring `base` with more colours:
    vertices in colour classes >= k are recoloured uniformly at random into 0..k-1."""
    col = np.array(base, dtype=np.int64)
    mask = col >= k
    col[mask] = [rng.randrange(k) for _ in range(int(mask.sum()))]
    return col


def fixed_k_search(n, adj_idx, k, col, max_iters, rng, select_move, tenure, deadline=None):
    """Tabu search for a conflict-free k-colouring starting from `col` (entries in 0..k-1).

    Runs at most `max_iters` iterations (None = unlimited) and stops early at `deadline`
    (perf_counter seconds, None = none). Each iteration asks `select_move` for one
    (vertex, new_colour), applies it, and forbids reverting it for `tenure(...)` iterations.

    Returns (best_col, best_conflicts, solved_at, iterations_run):
        best_col        numpy array, the colouring with the fewest conflicting edges seen
        best_conflicts  that number of conflicting edges (0 = legal k-colouring)
        solved_at       iteration at which conflicts first reached 0, or None
        iterations_run  how many iterations were executed
    """
    gamma = np.zeros((n, k), dtype=np.int32)      # gamma[v, c] = neighbours of v with colour c
    for v in range(n):
        np.add.at(gamma[v], col[adj_idx[v]], 1)
    rows = np.arange(n)
    conflicts = int(gamma[rows, col].sum() // 2)   # each conflicting edge counted from both ends
    best_col, best_conf = col.copy(), conflicts
    solved_at = 0 if conflicts == 0 else None
    tabu_until = np.zeros((n, k), dtype=np.int64)  # (v, c) is tabu while tabu_until[v, c] > it
    it = 0
    while best_conf > 0:
        if max_iters is not None and it >= max_iters:
            break
        if deadline is not None and (it & 63) == 0 and time.perf_counter() > deadline:
            break
        it += 1
        own = gamma[rows, col]
        conflicting = np.flatnonzero(own > 0)       # vertices currently in a conflict
        v, c = select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng)
        v, c = int(v), int(c)
        old = int(col[v])
        if not (0 <= v < n) or not (0 <= c < k) or c == old:
            raise ValueError(f"select_move returned an invalid move (v={v}, c={c}, current={old}, k={k})")
        conflicts += int(gamma[v, c]) - int(gamma[v, old])
        col[v] = c
        nb = adj_idx[v]
        gamma[nb, old] -= 1
        gamma[nb, c] += 1
        tabu_until[v, old] = it + int(tenure(n, k, int(conflicting.size), it, rng))
        if conflicts < best_conf:
            best_conf = conflicts
            best_col = col.copy()
            if conflicts == 0:
                solved_at = it
    return best_col, best_conf, solved_at, it


def color_graph(n, edges, time_limit, select_move, tenure, seed=12345):
    """Decreasing-k wrapper for held-out evaluation: DSATUR, then repeatedly search for a
    legal colouring with one colour fewer until the time budget runs out."""
    start = time.perf_counter()
    deadline = start + 0.92 * time_limit
    rng = random.Random(seed)
    adj_idx = build_adjacency(n, edges)
    best = dsatur(n, adj_idx)
    k = max(best) + 1
    while k > 1 and time.perf_counter() < deadline:
        col = start_coloring(best, k - 1, rng)
        best_col, conf, _, _ = fixed_k_search(n, adj_idx, k - 1, col, None, rng, select_move, tenure, deadline)
        if conf == 0:
            best = best_col.tolist()
            k -= 1
        else:
            break
    return best
