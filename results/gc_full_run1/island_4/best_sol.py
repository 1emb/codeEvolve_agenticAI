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
    """Novel hybrid evolutionary graph coloring algorithm"""
    start = time.perf_counter()
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Degree calculation
    degree = [len(a) for a in adj]
    
    # Count conflicts in coloring
    def count_conflicts(color):
        conflicts = 0
        for u in range(n):
            for v in adj[u]:
                if color[u] == color[v]:
                    conflicts += 1
        return conflicts // 2
    
    # Simple evolutionary algorithm with novel features
    def evolutionary_coloring():
        # Population size and parameters
        pop_size = min(30, max(10, n // 4))
        max_generations = 500
        mutation_rate = 0.1
        
        # Generate initial population with multiple strategies
        def generate_individual():
            # Strategy 1: DSATUR-based coloring
            color = [-1] * n
            seen = [set() for _ in range(n)]
            uncolored = set(range(n))
            
            while uncolored:
                candidates = list(uncolored)
                scores = []
                for v in candidates:
                    sat_degree = len(seen[v])
                    # Mix of DSATUR and random tie-breaking
                    score = (sat_degree, degree[v], random.random() * 0.01)
                    scores.append(score)
                
                best_idx = scores.index(max(scores))
                v = candidates[best_idx]
                
                c = 0
                while c in seen[v]:
                    c += 1
                color[v] = c
                uncolored.remove(v)
                
                for w in adj[v]:
                    seen[w].add(c)
                    
                if time.perf_counter() - start > time_limit * 0.7:
                    break
                    
            return color
        
        # Evaluate fitness (conflicts + penalty for too many colors)
        def evaluate_fitness(color):
            conflicts = count_conflicts(color)
            num_colors = max(color) + 1 if color else 0
            # Penalty for excessive colors (to encourage minimal coloring)
            penalty = num_colors * 10
            return conflicts + penalty
        
        # Generate initial population
        population = []
        for _ in range(pop_size):
            individual = generate_individual()
            population.append(individual)
        
        # Sort by fitness
        population.sort(key=evaluate_fitness)
        best_solution = population[0][:]
        
        # Evolution loop
        for generation in range(max_generations):
            if time.perf_counter() - start > time_limit * 0.95:
                break
                
            # Selection: tournament selection
            new_population = []
            
            # Elitism: keep top 20%
            elite_count = max(1, pop_size // 5)
            new_population.extend(population[:elite_count])
            
            # Generate offspring through crossover and mutation
            while len(new_population) < pop_size:
                # Tournament selection
                parent1 = min(random.sample(population, 3), key=evaluate_fitness)
                parent2 = min(random.sample(population, 3), key=evaluate_fitness)
                
                # Crossover: uniform crossover on colors
                child = parent1[:]
                for i in range(n):
                    if random.random() < 0.5:
                        child[i] = parent2[i]
                
                # Mutation: random color change for some vertices
                for i in range(n):
                    if random.random() < mutation_rate:
                        # Find valid color (not used by neighbors)
                        forbidden = set()
                        for neighbor in adj[i]:
                            forbidden.add(child[neighbor])
                        
                        # Try to find a new color
                        new_color = random.randint(0, max(child) + 10)
                        while new_color in forbidden:
                            new_color += 1
                        child[i] = new_color
                
                new_population.append(child)
            
            population = new_population
            population.sort(key=evaluate_fitness)
            
            # Update best solution
            if evaluate_fitness(population[0]) < evaluate_fitness(best_solution):
                best_solution = population[0][:]
        
        return best_solution
    
    # Local improvement to refine solution
    def improve_solution(initial_color):
        color = initial_color[:]
        current_conflicts = count_conflicts(color)
        
        # Iterative improvement using vertex reassignment
        max_iterations = 5000
        iterations = 0
        
        while iterations < max_iterations:
            if time.perf_counter() - start > time_limit * 0.95:
                break
                
            # Pick a random vertex
            u = random.randint(0, n - 1)
            old_color = color[u]
            
            # Get forbidden colors (neighbors' colors)
            forbidden = set()
            for v in adj[u]:
                forbidden.add(color[v])
            
            # Try to find a better color for vertex u
            best_color = old_color
            best_improvement = 0
            
            # Try colors up to a reasonable bound
            for c in range(max(color) + 20):
                if c not in forbidden and c != old_color:
                    # Temporarily change color
                    color[u] = c
                    new_conflicts = count_conflicts(color)
                    improvement = current_conflicts - new_conflicts
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_color = c
                    
                    # Restore color for next test
                    color[u] = old_color
            
            # Apply the best move if found
            if best_improvement > 0:
                color[u] = best_color
                current_conflicts -= best_improvement
            else:
                # Small chance to escape local optima
                if random.random() < 0.01:
                    color[u] = random.randint(0, max(color) + 10)
            
            iterations += 1
        
        return color
    
    # Main algorithm
    try:
        # Phase 1: Evolutionary approach
        evolved_solution = evolutionary_coloring()
        
        # Phase 2: Local improvement
        final_solution = improve_solution(evolved_solution)
        
        # Phase 3: Final validation and cleanup
        conflicts = count_conflicts(final_solution)
        if conflicts > 0:
            # Apply final fix to ensure valid coloring
            for _ in range(1000):
                if time.perf_counter() - start > time_limit * 0.95:
                    break
                # Simple greedy fix: reassign vertices causing conflicts
                fixed_any = False
                for u in range(n):
                    # Check if this vertex creates conflicts
                    conflicts_with_u = sum(1 for v in adj[u] if final_solution[u] == final_solution[v])
                    if conflicts_with_u > 0:
                        # Try to reassign
                        forbidden = set()
                        for v in adj[u]:
                            forbidden.add(final_solution[v])
                        
                        # Find a new color
                        new_color = 0
                        while new_color in forbidden:
                            new_color += 1
                        final_solution[u] = new_color
                        fixed_any = True
                if not fixed_any:
                    break
        
    except Exception as e:
        # Fallback to simple DSATUR
        color = [-1] * n
        seen = [set() for _ in range(n)]
        uncolored = set(range(n))
        
        while uncolored:
            candidates = list(uncolored)
            scores = []
            for v in candidates:
                sat_degree = len(seen[v])
                score = (sat_degree, degree[v], random.random())
                scores.append(score)
            
            best_idx = scores.index(max(scores))
            v = candidates[best_idx]
            
            c = 0
            while c in seen[v]:
                c += 1
            color[v] = c
            uncolored.remove(v)
            
            for w in adj[v]:
                seen[w].add(c)
                
            if time.perf_counter() - start > time_limit * 0.8:
                break
                
        final_solution = color
    
    return final_solution
# EVOLVE-BLOCK-END
