# ===--------------------------------------------------------------------------------------===#
#
# Initial solution for the graph coloring problem: DSATUR.
#
# Contract (do not change the function name or signature):
#   color_graph(n, edges, time_limit) -> list[int]
#     n          number of vertices, labelled 0 .. n-1
#     edges      list of (u, v) tuples, undirected, 0-indexed
#     time_limit wall-clock budget in seconds; the function must return before it elapses
#     returns    list of n integers, entry i is the color of vertex i (any integer labels)
#
# ===--------------------------------------------------------------------------------------===#

import time


# EVOLVE-BLOCK-START
def color_graph(n, edges, time_limit):
    """DSATUR: repeatedly color the uncolored vertex that sees the most distinct
    neighbour colors (ties broken by degree) with the smallest color not used by
    any of its neighbours."""
    start = time.perf_counter()

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    color = [-1] * n
    seen = [set() for _ in range(n)]  # colors already used on each vertex's neighbours
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
        if time.perf_counter() - start > time_limit * 0.95:
            break

    # safety: any vertex left uncolored gets a fresh color
    nxt = max(color) + 1
    for v in range(n):
        if color[v] < 0:
            color[v] = nxt
            nxt += 1
    return color
# EVOLVE-BLOCK-END
