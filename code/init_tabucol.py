# ===--------------------------------------------------------------------------------------===#
#
# Initial solution for the graph coloring problem: DSATUR + TabuCol with decreasing k.
#
# Contract (do not change the function name or signature):
#   color_graph(n, edges, time_limit) -> list[int]
#     n          number of vertices, labelled 0 .. n-1
#     edges      list of (u, v) tuples, undirected, 0-indexed
#     time_limit wall-clock budget in seconds; the function must return before it elapses
#     returns    list of n integers, entry i is the color of vertex i (any integer labels)
#
# ===--------------------------------------------------------------------------------------===#

import random
import time

import numpy as np


# EVOLVE-BLOCK-START
def dsatur(n, adj):
    """Greedy DSATUR coloring: pick the uncolored vertex that sees the most distinct
    neighbour colors (ties by degree), give it the smallest color its neighbours lack."""
    color = [-1] * n
    seen = [set() for _ in range(n)]
    degree = [len(a) for a in adj]
    uncolored = set(range(n))
    while uncolored:
        v = max(uncolored, key=lambda x: (len(seen[x]), degree[x]))
        c = 0
        while c in seen[v]:
            c += 1
        color[v] = c
        uncolored.remove(v)
        for w in adj[v]:
            seen[w].add(c)
    return color


def tabucol(n, adj_idx, k, col, deadline, rng):
    """TabuCol: search for a proper k-coloring starting from `col` (entries in 0..k-1),
    minimising the number of conflicting edges with a tabu search on 1-vertex recolorings.
    Returns (coloring, conflicts) for the best coloring found before `deadline`."""
    # gamma[v, c] = number of neighbours of v having color c
    gamma = np.zeros((n, k), dtype=np.int32)
    for v in range(n):
        np.add.at(gamma[v], col[adj_idx[v]], 1)
    conflicts = int(sum(gamma[v, col[v]] for v in range(n)) // 2)
    best_col, best_conf = col.copy(), conflicts
    tabu = np.zeros((n, k), dtype=np.int64)   # iteration until which (v, c) is tabu
    rows = np.arange(n)
    it = 0
    while best_conf > 0:
        it += 1
        if it % 64 == 0 and time.perf_counter() > deadline:
            break
        own = gamma[rows, col]                       # conflicts each vertex currently causes
        conf_v = np.flatnonzero(own > 0)
        if conf_v.size == 0:
            break
        delta = gamma[conf_v] - own[conf_v, None]    # change in conflicts for each move
        # tabu moves are forbidden unless they lead to a new best (aspiration)
        allowed = tabu[conf_v] <= it
        aspir = (conflicts + delta) < best_conf
        delta_masked = np.where(allowed | aspir, delta, 1 << 30)
        delta_masked[np.arange(conf_v.size), col[conf_v]] = 1 << 30   # staying is not a move
        flat = int(np.argmin(delta_masked))
        i, c = divmod(flat, k)
        d = int(delta_masked[i, c])
        if d >= (1 << 30):            # everything tabu: pick a random move
            i = rng.randrange(conf_v.size)
            c = rng.randrange(k)
            if c == col[conf_v[i]]:
                c = (c + 1) % k
            d = int(delta[i, c])
        v = int(conf_v[i])
        old = int(col[v])
        col[v] = c
        nb = adj_idx[v]
        gamma[nb, old] -= 1
        gamma[nb, c] += 1
        conflicts += d
        tabu[v, old] = it + int(0.6 * conf_v.size) + rng.randrange(10)
        if conflicts < best_conf:
            best_conf = conflicts
            best_col = col.copy()
    return best_col, best_conf


def color_graph(n, edges, time_limit):
    """DSATUR gives a legal coloring with k0 colors; then repeatedly try to find a legal
    coloring with one color fewer using TabuCol, until the time budget runs out."""
    start = time.perf_counter()
    deadline = start + time_limit * 0.92
    rng = random.Random(12345)

    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    adj_idx = [np.fromiter(a, dtype=np.int64, count=len(a)) for a in adj]

    best = dsatur(n, adj)
    k = max(best) + 1
    while k > 1 and time.perf_counter() < deadline:
        target = k - 1
        # start from the best legal coloring, recolor the removed class randomly
        col = np.array(best, dtype=np.int64)
        col[col >= target] = [rng.randrange(target) for _ in range(int((col >= target).sum()))]
        col, conf = tabucol(n, adj_idx, target, col, deadline, rng)
        if conf == 0:
            best = col.tolist()
            k = target
        else:
            break            # could not find a (k-1)-coloring in the remaining time
    return best
# EVOLVE-BLOCK-END
