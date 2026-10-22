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
    """Enhanced hybrid graph coloring with better time utilization and local search"""
    start = time.perf_counter()
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute degrees for ordering
    degree = [len(a) for a in adj]
    
    # More sophisticated DSATUR variant with multiple strategies
    def smart_dsatur():
        color = [-1] * n
        seen = [set() for _ in range(n)]  # colors already used on each vertex's neighbours
        uncolored = set(range(n))
        
        while uncolored:
            # Select vertex with highest DSATUR score
            candidates = list(uncolored)
            if not candidates:
                break
                
            # DSATUR score with multiple tie-breaking factors
            def dsatur_score(x):
                distinct_colors = len(seen[x])
                deg = degree[x]
                # Add slight randomness to break ties but keep it predictable
                random_factor = random.random() * 0.001
                return (distinct_colors, deg, random_factor)
            
            v = max(candidates, key=dsatur_score)
            
            # Assign smallest available color
            c = 0
            while c in seen[v]:
                c += 1
            color[v] = c
            
            # Update seen colors for neighbors
            uncolored.remove(v)
            for w in adj[v]:
                seen[w].add(c)
            
            # Periodic time check
            if time.perf_counter() - start > time_limit * 0.8:
                break
        
        return color
    
    # Try multiple construction strategies
    best_coloring = None
    best_num_colors = float('inf')
    
    # Run several construction attempts
    attempts = 0
    while attempts < 10 and time.perf_counter() - start < time_limit * 0.8:
        color = smart_dsatur()
        num_colors = max(color) + 1
        if num_colors < best_num_colors:
            best_num_colors = num_colors
            best_coloring = color[:]
        attempts += 1
    
    # Fallback to simple greedy if nothing found
    if best_coloring is None:
        best_coloring = [0] * n
        for v in range(n):
            used_colors = set()
            for neighbor in adj[v]:
                if best_coloring[neighbor] != -1:
                    used_colors.add(best_coloring[neighbor])
            
            c = 0
            while c in used_colors:
                c += 1
            best_coloring[v] = c
    
    # Comprehensive local search improvement with better conflict handling
    def advanced_local_search():
        nonlocal best_coloring
        color = best_coloring[:]
        
        def count_conflicts():
            conflicts = 0
            for v in range(n):
                for neighbor in adj[v]:
                    if color[v] == color[neighbor]:
                        conflicts += 1
            return conflicts // 2
        
        def get_conflicted_vertices():
            conflicted = []
            for v in range(n):
                for neighbor in adj[v]:
                    if color[v] == color[neighbor]:
                        conflicted.append(v)
                        break
            return conflicted
        
        # Multi-phase local search
        max_iter = 50000
        best_conflicts = count_conflicts()
        best_solution = color[:]
        
        for iteration in range(max_iter):
            if time.perf_counter() - start > time_limit * 0.95:
                break
                
            # Get conflicted vertices
            conflicted = get_conflicted_vertices()
            if not conflicted:
                break
            
            # Phase 1: Try to fix a conflicted vertex with a better color
            v = random.choice(conflicted)
            old_color = color[v]
            
            # Get neighbor colors
            neighbor_colors = set()
            for neighbor in adj[v]:
                if color[neighbor] != -1:
                    neighbor_colors.add(color[neighbor])
            
            # Try to assign a smaller color that's not used by neighbors
            new_color = 0
            while new_color in neighbor_colors and new_color <= max(color):
                new_color += 1
            
            # If found a better color, apply it
            if new_color < old_color:
                color[v] = new_color
                new_conflicts = count_conflicts()
                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best_solution = color[:]
            else:
                # Phase 2: Try to reduce overall number of colors
                # Look for vertices that can be recolored with smaller colors
                for v_test in range(n):
                    if v_test not in conflicted:  # Skip conflicted ones for now
                        old_c = color[v_test]
                        neighbor_colors = set()
                        for neighbor in adj[v_test]:
                            if color[neighbor] != -1:
                                neighbor_colors.add(color[neighbor])
                        
                        # Try to find a smaller color
                        new_c = 0
                        while new_c in neighbor_colors and new_c <= max(color):
                            new_c += 1
                        
                        if new_c < old_c:
                            color[v_test] = new_c
                            new_conflicts = count_conflicts()
                            if new_conflicts < best_conflicts:
                                best_conflicts = new_conflicts
                                best_solution = color[:]
                                break  # Found improvement, restart
    
        return best_solution
    
    # Apply advanced local search
    final_coloring = advanced_local_search()
    
    # Final validation to ensure no conflicts
    def validate_coloring(c):
        for v in range(n):
            for neighbor in adj[v]:
                if c[v] == c[neighbor]:
                    return False
        return True
    
    # Ensure final coloring is valid
    if not validate_coloring(final_coloring):
        # If still invalid, use a final greedy approach
        final_coloring = [0] * n
        for v in range(n):
            used_colors = set()
            for neighbor in adj[v]:
                if final_coloring[neighbor] != -1:
                    used_colors.add(final_coloring[neighbor])
            
            c = 0
            while c in used_colors:
                c += 1
            final_coloring[v] = c
    
    return final_coloring
# EVOLVE-BLOCK-END
