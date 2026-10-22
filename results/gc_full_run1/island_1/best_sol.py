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
    """Hybrid population-based graph coloring using multiple strategies"""
    start = time.perf_counter()
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute degrees
    degree = [len(a) for a in adj]
    
    # Initialize population with diverse strategies
    population_size = min(20, max(5, n // 50))
    population = []
    best_solution = None
    best_k = float('inf')
    
    # Generate initial diverse solutions using different heuristics
    for _ in range(population_size):
        if time.perf_counter() - start > time_limit * 0.1:
            break
            
        # Mix of different construction heuristics
        strategy = random.randint(0, 3)
        solution = None
        
        if strategy == 0:
            # DSATUR variant with randomization
            solution = construct_dsatur(n, adj, degree, randomize=True)
        elif strategy == 1:
            # Largest Degree First with random tie-breaking
            solution = construct_ldf(n, adj, degree, randomize=True)
        elif strategy == 2:
            # Random ordering
            solution = construct_random(n, adj, degree)
        else:
            # Color greedy with random vertex ordering
            solution = construct_greedy_random(n, adj, degree)
        
        k = max(solution) + 1
        if k < best_k:
            best_k = k
            best_solution = solution[:]
        population.append(solution)
    
    # If we didn't get enough solutions, fill with copies of best
    while len(population) < population_size:
        population.append(best_solution[:])
    
    # Main evolutionary search
    tabu_list = []  # Tabu tenure for recently changed colors
    tabu_tenure = max(5, n // 20)
    
    # Track best overall solution
    best_overall = best_solution[:]
    best_overall_k = best_k
    
    # Evolution parameters
    generations = 0
    stagnation_count = 0
    max_stagnation = 50
    
    while time.perf_counter() - start < time_limit * 0.9:
        generations += 1
        new_population = []
        
        # Selection and crossover
        for _ in range(population_size):
            if time.perf_counter() - start > time_limit * 0.95:
                break
                
            # Tournament selection
            parent1 = tournament_select(population, 3)
            parent2 = tournament_select(population, 3)
            
            # Crossover (order-based)
            child = crossover(parent1, parent2, n, adj)
            
            # Local improvement with tabu search
            improved_child = local_improve_tabu(child, n, adj, tabu_list, tabu_tenure)
            
            # Mutation (random color changes)
            if random.random() < 0.3:
                improved_child = mutate(improved_child, n, adj)
            
            new_population.append(improved_child)
        
        # Elitism: keep best from old and new populations
        all_solutions = population + new_population
        all_solutions.sort(key=lambda sol: max(sol) + 1)
        
        # Keep top solutions
        population = all_solutions[:population_size]
        
        # Update best overall
        current_best = population[0]
        current_k = max(current_best) + 1
        if current_k < best_overall_k:
            best_overall = current_best[:]
            best_overall_k = current_k
            stagnation_count = 0
        else:
            stagnation_count += 1
            
        # Reset stagnation if significant improvement
        if stagnation_count > max_stagnation:
            # Restart with new diverse solutions
            population = []
            for _ in range(population_size):
                strategy = random.randint(0, 3)
                if strategy == 0:
                    solution = construct_dsatur(n, adj, degree, randomize=True)
                elif strategy == 1:
                    solution = construct_ldf(n, adj, degree, randomize=True)
                elif strategy == 2:
                    solution = construct_random(n, adj, degree)
                else:
                    solution = construct_greedy_random(n, adj, degree)
                population.append(solution)
            stagnation_count = 0
            
        if time.perf_counter() - start > time_limit * 0.95:
            break
    
    # Final improvement of best solution
    final_solution = local_improve_tabu(best_overall, n, adj, tabu_list, tabu_tenure)
    return final_solution


def construct_dsatur(n, adj, degree, randomize=False):
    """Construct solution using DSATUR heuristic with optional randomization"""
    color = [-1] * n
    seen = [set() for _ in range(n)]
    uncolored = set(range(n))
    
    while uncolored:
        if randomize:
            # Add small random noise to make ties less likely
            candidates = [(v, (len(seen[v]), degree[v] + random.uniform(0, 0.1))) 
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


def construct_ldf(n, adj, degree, randomize=False):
    """Construct solution using Largest Degree First heuristic"""
    color = [-1] * n
    seen = [set() for _ in range(n)]
    
    # Sort vertices by degree (descending)
    vertices = list(range(n))
    if randomize:
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


def construct_greedy_random(n, adj, degree):
    """Construct solution with greedy coloring on random order"""
    color = [-1] * n
    seen = [set() for _ in range(n)]
    
    # Create random order with bias toward higher degree vertices
    vertices = list(range(n))
    random.shuffle(vertices)
    
    # Sort by degree in descending order with some randomization
    vertices.sort(key=lambda x: (degree[x], random.random()), reverse=True)
    
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


def tournament_select(population, k):
    """Select individual from population using tournament selection"""
    tournament = random.sample(population, min(k, len(population)))
    return min(tournament, key=lambda sol: max(sol) + 1)


def crossover(parent1, parent2, n, adj):
    """Order-based crossover between two solutions"""
    child = [-1] * n
    # Take half of vertices from parent1, half from parent2
    half = n // 2
    selected = set(random.sample(range(n), half))
    
    # Copy from parent1 first
    for i in range(n):
        if i in selected:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    
    # Fix conflicts using greedy approach
    return fix_conflicts(child, n, adj)


def fix_conflicts(coloring, n, adj):
    """Fix conflicts in coloring using greedy approach"""
    seen = [set() for _ in range(n)]
    # Precompute what colors are used by neighbors
    for i in range(n):
        if coloring[i] != -1:
            for neighbor in adj[i]:
                if coloring[neighbor] != -1:
                    seen[i].add(coloring[neighbor])
    
    # Re-color vertices in order of conflict priority
    vertices = list(range(n))
    vertices.sort(key=lambda x: len(adj[x]), reverse=True)  # prioritize high-degree
    
    for v in vertices:
        if coloring[v] == -1:
            continue
        # Find smallest available color
        c = 0
        while c in seen[v]:
            c += 1
        coloring[v] = c
        # Update neighbor seen sets
        for w in adj[v]:
            seen[w].add(c)
    
    return coloring


def local_improve_tabu(coloring, n, adj, tabu_list, tabu_tenure):
    """Local improvement with tabu search"""
    # Create a copy to work with
    solution = coloring[:]
    seen = [set() for _ in range(n)]
    
    # Initialize seen sets
    for i in range(n):
        for neighbor in adj[i]:
            if solution[neighbor] != -1:
                seen[i].add(solution[neighbor])
    
    improved = True
    iterations = 0
    max_iterations = 100
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try to improve each vertex
        for v in range(n):
            if time.perf_counter() - time.perf_counter() > 0.01:  # Safety check
                break
                
            current_color = solution[v]
            # Find best alternative color
            best_color = current_color
            best_delta = 0  # delta = number of conflicts removed
            
            # Check all possible color changes
            for c in range(max(solution) + 10):  # Look ahead a bit
                if c in seen[v] or (c, v) in tabu_list:
                    continue
                    
                # Count conflicts we would eliminate by changing to color c
                delta = 0
                # Check neighbors that would be affected
                for neighbor in adj[v]:
                    if solution[neighbor] == current_color:
                        delta += 1
                    elif solution[neighbor] == c:
                        delta -= 1
                
                if delta > best_delta:
                    best_delta = delta
                    best_color = c
            
            # Apply improvement if beneficial
            if best_color != current_color and best_delta > 0:
                # Update tabu list
                if len(tabu_list) >= tabu_tenure:
                    tabu_list.pop(0)
                tabu_list.append((current_color, v))
                
                # Apply change
                solution[v] = best_color
                # Update seen sets
                for neighbor in adj[v]:
                    seen[neighbor].discard(current_color)
                    seen[neighbor].add(best_color)
                
                improved = True
    
    return solution


def mutate(coloring, n, adj):
    """Random mutation of coloring"""
    solution = coloring[:]
    # Change a few random vertices to different colors
    num_changes = max(1, n // 50)
    for _ in range(num_changes):
        v = random.randint(0, n-1)
        # Find smallest available color
        seen_colors = set()
        for neighbor in adj[v]:
            if solution[neighbor] != -1:
                seen_colors.add(solution[neighbor])
        
        c = 0
        while c in seen_colors:
            c += 1
        solution[v] = c
    
    return solution
# EVOLVE-BLOCK-END
