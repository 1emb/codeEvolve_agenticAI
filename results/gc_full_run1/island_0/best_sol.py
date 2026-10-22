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
    """Enhanced hybrid approach: diversification + advanced local improvement"""
    start = time.perf_counter()
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute degrees
    degree = [len(a) for a in adj]
    
    # Multiple diversified starting points
    best_solution = None
    best_k = float('inf')
    
    # Generate several diverse initial solutions
    num_starts = min(20, max(5, n // 30))
    
    for i in range(num_starts):
        if time.perf_counter() - start > time_limit * 0.7:
            break
            
        # Different construction strategies
        if i == 0:
            # Standard DSATUR
            solution = construct_dsatur(n, adj, degree)
        elif i == 1:
            # DSATUR with random tie-breaking
            solution = construct_dsatur(n, adj, degree, randomize=True)
        elif i == 2:
            # Largest Degree First
            solution = construct_ldf(n, adj, degree)
        elif i == 3:
            # Random ordering
            solution = construct_random(n, adj, degree)
        elif i == 4:
            # Randomized DSATUR with higher randomness
            solution = construct_dsatur(n, adj, degree, randomize=True, noise=0.5)
        else:
            # Hybrid: mix LDF with some randomization
            solution = construct_hybrid(n, adj, degree)
        
        # Local improvement with more sophisticated approach
        improved_solution = local_improve_advanced(solution, n, adj, start, time_limit)
        
        k = max(improved_solution) + 1
        if k < best_k:
            best_k = k
            best_solution = improved_solution[:]
    
    # If we have no solution, fall back to basic DSATUR
    if best_solution is None:
        best_solution = construct_dsatur(n, adj, degree)
    
    # Final comprehensive local search
    final_solution = local_improve_comprehensive(best_solution, n, adj, start, time_limit)
    
    return final_solution


def construct_dsatur(n, adj, degree, randomize=False, noise=0.1):
    """Construct solution using DSATUR heuristic"""
    color = [-1] * n
    seen = [set() for _ in range(n)]
    uncolored = set(range(n))
    
    while uncolored:
        if randomize:
            # Add noise to make ties less likely
            candidates = [(v, (len(seen[v]), degree[v] + random.uniform(0, noise))) 
                         for v in uncolored]
            v = max(candidates, key=lambda x: x[1])[0]
        else:
            v = max(uncolored, key=lambda x: (len(seen[x]), degree[x]))
            
        c = 0
        while c in seen[v]:
            c += 1
        color[v] = c
        uncolored.remove(v)
        for w in adj[v]:
            seen[w].add(c)
    
    return color


def construct_ldf(n, adj, degree):
    """Construct solution using Largest Degree First heuristic"""
    color = [-1] * n
    seen = [set() for _ in range(n)]
    
    # Sort vertices by degree (descending)
    vertices = list(range(n))
    vertices.sort(key=lambda x: degree[x], reverse=True)
    
    for v in vertices:
        if color[v] != -1:
            continue
        c = 0
        while c in seen[v]:
            c += 1
        color[v] = c
        for w in adj[v]:
            seen[w].add(c)
    
    return color


def construct_random(n, adj, degree):
    """Construct solution with random vertex ordering"""
    color = [-1] * n
    seen = [set() for _ in range(n)]
    
    vertices = list(range(n))
    random.shuffle(vertices)
    
    for v in vertices:
        if color[v] != -1:
            continue
        c = 0
        while c in seen[v]:
            c += 1
        color[v] = c
        for w in adj[v]:
            seen[w].add(c)
    
    return color


def construct_hybrid(n, adj, degree):
    """Construct solution with hybrid approach"""
    color = [-1] * n
    seen = [set() for _ in range(n)]
    
    # Start with LDF ordering but add some randomness
    vertices = list(range(n))
    random.shuffle(vertices)
    vertices.sort(key=lambda x: degree[x], reverse=True)
    
    for v in vertices:
        if color[v] != -1:
            continue
        c = 0
        while c in seen[v]:
            c += 1
        color[v] = c
        for w in adj[v]:
            seen[w].add(c)
    
    return color


def count_conflicts(colors, adj, n):
    """Count number of conflicts in current coloring"""
    conflicts = 0
    for u in range(n):
        for v in adj[u]:
            if colors[u] == colors[v]:
                conflicts += 1
    return conflicts // 2


def local_improve_advanced(coloring, n, adj, start_time, time_limit):
    """Advanced local improvement with more sophisticated neighborhood search"""
    # Create a copy to work with
    solution = coloring[:]
    improved = True
    max_iterations = int(100000 * time_limit / 10.0)  # More iterations for better search
    
    # Precompute neighbor colors for efficiency
    neighbor_colors = []
    for i in range(n):
        nc = set()
        for neighbor in adj[i]:
            if solution[neighbor] != -1:
                nc.add(solution[neighbor])
        neighbor_colors.append(nc)
    
    # Track best solution found
    best_solution = solution[:]
    best_conflicts = count_conflicts(solution, adj, n)
    
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try to improve each vertex - but do it in random order for better exploration
        vertices = list(range(n))
        random.shuffle(vertices)
        
        for v in vertices:
            if time.perf_counter() - start_time > time_limit * 0.95:
                return best_solution
                
            current_color = solution[v]
            
            # Find available colors efficiently
            used_colors = neighbor_colors[v]
            
            # Try to find a better color - look at both existing and new colors
            best_color = current_color
            best_improvement = 0
            
            # Try colors from 0 up to a reasonable bound
            # Be more systematic about trying colors
            max_color_to_try = max(solution) + 50  # Try more colors
            for c in range(max_color_to_try):
                if c in used_colors:
                    continue
                    
                # Count how many conflicts would be eliminated vs created
                conflicts_eliminated = 0
                conflicts_created = 0
                
                for neighbor in adj[v]:
                    if solution[neighbor] == current_color:
                        conflicts_eliminated += 1
                    elif solution[neighbor] == c:
                        conflicts_created += 1
                
                improvement = conflicts_eliminated - conflicts_created
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_color = c
            
            # Also consider moving to lower numbered colors if that's beneficial
            # (this helps reduce overall color count)
            if best_improvement > 0:
                solution[v] = best_color
                # Update neighbor colors
                for neighbor in adj[v]:
                    if solution[neighbor] == current_color:
                        neighbor_colors[neighbor].discard(current_color)
                        neighbor_colors[neighbor].add(best_color)
                    elif solution[neighbor] == best_color:
                        neighbor_colors[neighbor].discard(best_color)
                        neighbor_colors[neighbor].add(current_color)
                improved = True
                # Update best solution
                current_conflicts = count_conflicts(solution, adj, n)
                if current_conflicts < best_conflicts:
                    best_solution = solution[:]
                    best_conflicts = current_conflicts
            else:
                # Try to move to a lower color even if no immediate gain
                # to reduce overall color usage (more aggressive)
                if current_color > 0:
                    # Try a few lower colors
                    for c in range(min(current_color, max_color_to_try)):
                        if c not in used_colors:
                            old_color = solution[v]
                            solution[v] = c
                            new_conflicts = count_conflicts(solution, adj, n)
                            if new_conflicts < current_conflicts:
                                # Accept this change
                                for neighbor in adj[v]:
                                    if solution[neighbor] == old_color:
                                        neighbor_colors[neighbor].discard(old_color)
                                        neighbor_colors[neighbor].add(c)
                                    elif solution[neighbor] == c:
                                        neighbor_colors[neighbor].discard(c)
                                        neighbor_colors[neighbor].add(old_color)
                                improved = True
                                current_conflicts = new_conflicts
                                if current_conflicts < best_conflicts:
                                    best_solution = solution[:]
                                    best_conflicts = current_conflicts
                                break
                            else:
                                solution[v] = old_color  # Revert
    
    return best_solution


def local_improve_comprehensive(coloring, n, adj, start_time, time_limit):
    """Comprehensive local search with multiple strategies"""
    # Create a copy to work with
    solution = coloring[:]
    
    # Precompute neighbor colors for efficiency
    neighbor_colors = []
    for i in range(n):
        nc = set()
        for neighbor in adj[i]:
            if solution[neighbor] != -1:
                nc.add(solution[neighbor])
        neighbor_colors.append(nc)
    
    # Track best solution found
    best_solution = solution[:]
    best_conflicts = count_conflicts(solution, adj, n)
    
    # Multiple phases of improvement
    max_iterations = int(150000 * time_limit / 10.0)  # Even more iterations for final phase
    iterations = 0
    
    # Phase 1: Standard local search
    while iterations < max_iterations and time.perf_counter() - start_time < time_limit * 0.95:
        improved = False
        
        # Try to improve each vertex
        vertices = list(range(n))
        random.shuffle(vertices)
        
        for v in vertices:
            current_color = solution[v]
            used_colors = neighbor_colors[v]
            
            # Try to find a better color
            best_color = current_color
            best_improvement = 0
            
            # Try colors from 0 up to a reasonable bound
            max_color_to_try = max(solution) + 50  # Try more colors
            for c in range(max_color_to_try):
                if c in used_colors:
                    continue
                    
                # Count conflicts that would be eliminated vs created
                conflicts_eliminated = 0
                conflicts_created = 0
                
                for neighbor in adj[v]:
                    if solution[neighbor] == current_color:
                        conflicts_eliminated += 1
                    elif solution[neighbor] == c:
                        conflicts_created += 1
                
                improvement = conflicts_eliminated - conflicts_created
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_color = c
            
            if best_improvement > 0:
                solution[v] = best_color
                improved = True
                # Update neighbor colors
                for neighbor in adj[v]:
                    if solution[neighbor] == current_color:
                        neighbor_colors[neighbor].discard(current_color)
                        neighbor_colors[neighbor].add(best_color)
                    elif solution[neighbor] == best_color:
                        neighbor_colors[neighbor].discard(best_color)
                        neighbor_colors[neighbor].add(current_color)
                # Update best solution
                current_conflicts = count_conflicts(solution, adj, n)
                if current_conflicts < best_conflicts:
                    best_solution = solution[:]
                    best_conflicts = current_conflicts
            else:
                # Try more aggressive random perturbations to escape local optima
                if random.random() < 0.15:  # Higher probability for perturbations
                    v = random.randint(0, n - 1)
                    used_colors = neighbor_colors[v]
                    # Try a random available color from a broader range
                    for c in range(max(solution) + 20):
                        if c not in used_colors:
                            old_color = solution[v]
                            solution[v] = c
                            new_conflicts = count_conflicts(solution, adj, n)
                            if new_conflicts < best_conflicts:
                                # Accept improvement
                                for neighbor in adj[v]:
                                    if solution[neighbor] == old_color:
                                        neighbor_colors[neighbor].discard(old_color)
                                        neighbor_colors[neighbor].add(c)
                                    elif solution[neighbor] == c:
                                        neighbor_colors[neighbor].discard(c)
                                        neighbor_colors[neighbor].add(old_color)
                                best_solution = solution[:]
                                best_conflicts = new_conflicts
                                improved = True
                            else:
                                solution[v] = old_color
                            break
        iterations += 1
    
    return best_solution
# EVOLVE-BLOCK-END
