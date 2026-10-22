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
import time
import random

def color_graph(n, edges, time_limit):
    """Multi-start DSATUR with best solution tracking"""
    start = time.perf_counter()

    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Precompute degrees for efficiency
    degree = [len(a) for a in adj]
    
    best_coloring = None
    best_k = float('inf')
    
    # Run multiple DSATUR instances with different randomizations
    # Use more restarts for better chance of finding good solutions
    num_restarts = max(5, int(time_limit * 3))
    
    for _ in range(num_restarts):
        if time.perf_counter() - start > time_limit * 0.95:
            break
            
        color = [-1] * n
        seen = [set() for _ in range(n)]  # colors already used on each vertex's neighbours
        uncolored = set(range(n))

        # DSATUR with randomized vertex selection
        while uncolored:
            # Get candidates with highest saturation degree and degree
            candidates = []
            for v in uncolored:
                sat_degree = len(seen[v])
                deg = degree[v]
                # Add random tie-breaker to improve results on hard graphs
                candidates.append((sat_degree, deg, random.random(), v))
            
            # Sort by (saturation degree, degree, random tiebreaker)
            candidates.sort(reverse=True)
            
            # Pick the best candidate
            v = candidates[0][3]
            
            # Assign lowest available color
            c = 0
            while c in seen[v]:
                c += 1
            color[v] = c
            
            # Update neighbor information
            uncolored.remove(v)
            for w in adj[v]:
                seen[w].add(c)
                
            # Early exit check
            if time.perf_counter() - start > time_limit * 0.95:
                break

        # Safety: any vertex left uncolored gets a fresh color
        nxt = max(color) + 1
        for v in range(n):
            if color[v] < 0:
                color[v] = nxt
                nxt += 1
                
        # Keep track of best solution found so far
        current_k = max(color) + 1
        if current_k < best_k:
            best_k = current_k
            best_coloring = color[:]
    
    # Return the best solution found
    return best_coloring if best_coloring is not None else [-1] * n
# EVOLVE-BLOCK-END
