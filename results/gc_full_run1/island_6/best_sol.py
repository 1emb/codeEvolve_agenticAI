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
import numpy as np


def color_graph(n, edges, time_limit):
    """Enhanced hybrid graph coloring with improved DSATUR and local search"""
    start = time.perf_counter()
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Calculate degrees for tie-breaking
    degree = [len(a) for a in adj]
    
    def dsatur_coloring(seed=None):
        """DSATUR coloring with optimized tie-breaking"""
        if seed is not None:
            random.seed(seed)
            
        color = [-1] * n
        seen = [set() for _ in range(n)]  # colors used by neighbors
        uncolored = set(range(n))
        
        while uncolored:
            # Select vertex with highest DSATUR score (most colored neighbors)
            candidates = list(uncolored)
            scores = [(len(seen[v]), degree[v], v) for v in candidates]
            # Use a small random perturbation to break ties consistently
            random_perturb = [random.random() * 0.001 for _ in range(len(candidates))]
            scores = [(s[0], s[1] + random_perturb[i], s[2]) for i, s in enumerate(scores)]
            v = max(scores, key=lambda x: x[:2])[2]  # Max by (dsat, degree, vertex_id)
            
            # Assign smallest available color
            c = 0
            while c in seen[v]:
                c += 1
            color[v] = c
            
            # Update neighbors' seen set
            uncolored.remove(v)
            for w in adj[v]:
                seen[w].add(c)
                
            # Check time limit
            if time.perf_counter() - start > time_limit * 0.95:
                break
                
        return color
    
    def count_conflicts(coloring):
        """Count total conflicts efficiently"""
        conflicts = 0
        for u in range(n):
            for v in adj[u]:
                if coloring[u] == coloring[v]:
                    conflicts += 1
        return conflicts // 2
    
    def is_valid_coloring(coloring):
        """Check if coloring is valid"""
        for u in range(n):
            for v in adj[u]:
                if coloring[u] == coloring[v]:
                    return False
        return True
    
    def local_refinement(initial_coloring):
        """Enhanced local refinement focusing on conflict reduction"""
        coloring = initial_coloring[:]
        max_iterations = 500
        iteration = 0
        
        while iteration < max_iterations:
            if time.perf_counter() - start > time_limit * 0.95:
                break
                
            conflicts = count_conflicts(coloring)
            if conflicts == 0:
                break
                
            # Focus on vertices with conflicts
            conflicting_vertices = [v for v in range(n) if count_conflicts_for_vertex(v, coloring) > 0]
            
            if not conflicting_vertices:
                break
                
            # Try to improve one vertex at a time
            improved = False
            for v in conflicting_vertices:
                if time.perf_counter() - start > time_limit * 0.95:
                    break
                    
                # Get all colors used by neighbors
                neighbor_colors = {coloring[u] for u in adj[v]}
                
                # Find the best color that fixes this vertex's conflicts
                # Try colors from 0 upward to find first valid one
                new_color = 0
                while new_color in neighbor_colors:
                    new_color += 1
                
                # If we found a better color (lower than current), apply it
                if new_color < coloring[v]:
                    old_color = coloring[v]
                    coloring[v] = new_color
                    
                    # Check if this actually reduced conflicts
                    new_conflicts = count_conflicts(coloring)
                    if new_conflicts >= conflicts:
                        coloring[v] = old_color  # Revert if not beneficial
                    else:
                        improved = True
                        conflicts = new_conflicts
                        
            if not improved:
                # Random perturbation to escape local optima
                if random.random() < 0.3:
                    v = random.choice(conflicting_vertices)
                    neighbor_colors = {coloring[u] for u in adj[v]}
                    new_color = 0
                    while new_color in neighbor_colors:
                        new_color += 1
                    if new_color < coloring[v]:  # Only if beneficial
                        coloring[v] = new_color
                        # Check if beneficial
                        new_conflicts = count_conflicts(coloring)
                        if new_conflicts >= conflicts:
                            coloring[v] = old_color
                        
            iteration += 1
            
        return coloring
    
    def count_conflicts_for_vertex(v, coloring):
        """Count conflicts for a specific vertex"""
        conflicts = 0
        for u in adj[v]:
            if coloring[v] == coloring[u]:
                conflicts += 1
        return conflicts
    
    def greedy_coloring():
        """Standard greedy coloring with degree ordering"""
        color = [-1] * n
        used_colors = [set() for _ in range(n)]
        
        # Color vertices in order of decreasing degree
        vertices = sorted(range(n), key=lambda x: degree[x], reverse=True)
        
        for v in vertices:
            # Find first unused color among neighbors
            forbidden = used_colors[v]
            c = 0
            while c in forbidden:
                c += 1
            color[v] = c
            
            # Mark this color as used by neighbors
            for u in adj[v]:
                used_colors[u].add(c)
                
        return color
    
    def multi_start_dsatur():
        """Run DSATUR multiple times with different seeds and return best"""
        best_result = None
        best_k = float('inf')
        best_conflicts = float('inf')
        
        # Adjust number of starts based on graph size
        num_starts = min(50, max(10, n // 10))
        
        for i in range(num_starts):
            if time.perf_counter() - start > time_limit * 0.95:
                break
                
            seed = i * 1000 + 1
            try:
                # Run DSATUR
                coloring = dsatur_coloring(seed)
                
                # Validate and refine
                if is_valid_coloring(coloring):
                    k = max(coloring) + 1
                    conflicts = count_conflicts(coloring)
                else:
                    # If invalid, do local refinement
                    coloring = local_refinement(coloring)
                    k = max(coloring) + 1
                    conflicts = count_conflicts(coloring)
                
                # Keep the best solution so far
                if conflicts < best_conflicts or (conflicts == best_conflicts and k < best_k):
                    best_conflicts = conflicts
                    best_k = k
                    best_result = coloring[:]
                    
            except Exception:
                continue
        
        # If we didn't get a good result, fall back to greedy
        if best_result is None:
            best_result = greedy_coloring()
            
        return best_result
    
    # Main algorithm
    # Try multi-start DSATUR approach
    result = multi_start_dsatur()
    
    # Final local refinement
    if result is not None:
        final_result = local_refinement(result)
        # Double-check validity
        if is_valid_coloring(final_result):
            return final_result
        else:
            # If somehow invalid, return original result
            return result
    
    # Fallback to greedy
    return greedy_coloring()
# EVOLVE-BLOCK-END
