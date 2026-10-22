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
    """Genetic algorithm with adaptive parameters for graph coloring"""
    start = time.perf_counter()
    
    # Build adjacency list representation
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute graph statistics
    degree = [len(a) for a in adj]
    avg_degree = sum(degree) / n if n > 0 else 0
    density = avg_degree / (n - 1) if n > 1 else 0
    
    # Initialize population
    population_size = min(100, max(20, n // 2))
    num_generations = 500
    
    # Create initial population with multiple strategies
    def create_individual():
        """Create an individual using different initialization strategies"""
        strategy = random.randint(0, 2)
        individual = [-1] * n
        
        if strategy == 0:
            # Random coloring
            for i in range(n):
                individual[i] = random.randint(0, n)
        elif strategy == 1:
            # Greedy coloring with random perturbations
            color = [-1] * n
            used_colors = [set() for _ in range(n)]
            
            # Color vertices in random order
            vertex_order = list(range(n))
            random.shuffle(vertex_order)
            
            for v in vertex_order:
                neighbor_colors = {color[u] for u in adj[v] if color[u] != -1}
                c = 0
                while c in neighbor_colors:
                    c += 1
                color[v] = c
            
            # Add some random perturbations
            individual = color[:]
            for _ in range(n // 10):
                if random.random() < 0.3:
                    v = random.randint(0, n-1)
                    neighbor_colors = {individual[u] for u in adj[v]}
                    c = 0
                    while c in neighbor_colors:
                        c += 1
                    individual[v] = c
        else:
            # DSATUR-like with randomization
            color = [-1] * n
            seen = [set() for _ in range(n)]
            uncolored = set(range(n))
            
            while uncolored:
                if len(uncolored) == 0:
                    break
                # Select vertex with highest saturation degree, breaking ties randomly
                candidates = list(uncolored)
                random.shuffle(candidates)
                v = max(candidates, key=lambda x: (len(seen[x]), degree[x]))
                
                c = 0
                while c in seen[v]:
                    c += 1
                color[v] = c
                uncolored.remove(v)
                for w in adj[v]:
                    seen[w].add(c)
            
            individual = color[:]
        
        return individual
    
    # Create initial population
    population = [create_individual() for _ in range(population_size)]
    
    def count_conflicts(individual):
        """Count number of conflicts in coloring"""
        conflicts = 0
        for u in range(n):
            for v in adj[u]:
                if individual[u] == individual[v]:
                    conflicts += 1
        return conflicts // 2
    
    def fitness(individual):
        """Fitness function: lower is better (number of conflicts)"""
        conflicts = count_conflicts(individual)
        # Add penalty for high color usage to encourage fewer colors
        max_color = max(individual) if individual else 0
        return conflicts + max_color * 0.1
    
    # Adaptive parameters based on graph characteristics
    mutation_rate = max(0.01, 0.1 - density * 0.05)  # Lower mutation rate for dense graphs
    crossover_rate = 0.8
    elite_size = population_size // 4
    
    # Main genetic algorithm loop
    best_solution = None
    best_fitness = float('inf')
    
    generation = 0
    while generation < num_generations:
        if time.perf_counter() - start > time_limit * 0.95:
            break
            
        # Evaluate fitness of entire population
        fitness_scores = [(individual, fitness(individual)) for individual in population]
        fitness_scores.sort(key=lambda x: x[1])
        
        # Update best solution
        current_best = fitness_scores[0][0]
        current_fitness = fitness_scores[0][1]
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = current_best[:]
        
        # Selection: tournament selection
        def tournament_selection(k=3):
            tournament = random.sample(fitness_scores, min(k, len(fitness_scores)))
            tournament.sort(key=lambda x: x[1])
            return tournament[0][0]
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(elite_size):
            new_population.append(fitness_scores[i][0][:])
        
        # Generate offspring
        while len(new_population) < population_size:
            if random.random() < crossover_rate:
                # Crossover
                parent1 = tournament_selection()
                parent2 = tournament_selection()
                
                # Uniform crossover
                child = [-1] * n
                for i in range(n):
                    if random.random() < 0.5:
                        child[i] = parent1[i]
                    else:
                        child[i] = parent2[i]
                
                # Mutation
                if random.random() < mutation_rate:
                    # Randomly mutate a vertex
                    vertex = random.randint(0, n-1)
                    neighbor_colors = {child[u] for u in adj[vertex]}
                    color = 0
                    while color in neighbor_colors:
                        color += 1
                    child[vertex] = color
                
                new_population.append(child)
            else:
                # Clone a parent and mutate
                parent = tournament_selection()
                child = parent[:]
                
                if random.random() < mutation_rate:
                    # Randomly mutate a vertex
                    vertex = random.randint(0, n-1)
                    neighbor_colors = {child[u] for u in adj[vertex]}
                    color = 0
                    while color in neighbor_colors:
                        color += 1
                    child[vertex] = color
                
                new_population.append(child)
        
        population = new_population[:population_size]
        generation += 1
    
    # Final local search improvement on the best solution
    if best_solution is not None:
        # Try to improve with local search
        def improve_local(current_solution):
            improved = True
            max_iterations = 1000
            iterations = 0
            
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                
                # Try to reduce conflicts by changing vertex colors
                for v in range(n):
                    if time.perf_counter() - start > time_limit * 0.95:
                        return current_solution
                        
                    current_color = current_solution[v]
                    neighbor_colors = {current_solution[u] for u in adj[v]}
                    
                    # Try to find a better color
                    best_color = current_color
                    best_conflicts = count_conflicts(current_solution)
                    
                    # Try all colors up to current max + 10
                    max_color = max(current_solution) + 10
                    for new_color in range(max_color):
                        if new_color not in neighbor_colors:
                            # Test if this change reduces conflicts
                            test_solution = current_solution[:]
                            test_solution[v] = new_color
                            new_conflicts = count_conflicts(test_solution)
                            
                            if new_conflicts < best_conflicts:
                                best_conflicts = new_conflicts
                                best_color = new_color
                                improved = True
                    
                    if best_color != current_color:
                        current_solution[v] = best_color
            
            return current_solution
        
        # Run final improvement
        best_solution = improve_local(best_solution)
    
    # Ensure all vertices are properly colored
    if best_solution is None:
        # Fallback to simple greedy coloring
        color = [-1] * n
        seen = [set() for _ in range(n)]
        uncolored = set(range(n))
        
        while uncolored:
            if time.perf_counter() - start > time_limit * 0.95:
                break
            v = max(uncolored, key=lambda x: (len(seen[x]), degree[x]))
            c = 0
            while c in seen[v]:
                c += 1
            color[v] = c
            uncolored.remove(v)
            for w in adj[v]:
                seen[w].add(c)
        
        # Safety check
        for v in range(n):
            if color[v] < 0:
                used_colors = {color[w] for w in adj[v] if color[w] != -1}
                c = 0
                while c in used_colors:
                    c += 1
                color[v] = c
        return color
    
    return best_solution
# EVOLVE-BLOCK-END
