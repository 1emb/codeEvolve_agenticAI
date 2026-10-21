# ===--------------------------------------------------------------------------------------===#
#
# Evolved decision rules for the fixed TabuCol framework (tabucol_framework.py).
#
# The framework runs this loop for a fixed number of iterations at a fixed number of colours k:
#     own         = gamma[rows, col]                     # conflicts each vertex is in
#     conflicting = vertices with own > 0
#     v, c        = select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng)
#     recolour v to c, update gamma / conflicts
#     tabu_until[v, old_colour] = it + tenure(n, k, len(conflicting), it, rng)
#     track the best (fewest conflicts) colouring seen
#
# Inputs to select_move:
#   gamma        int32 array (n, k): gamma[v, c] = number of neighbours of v that have colour c
#   col          int64 array (n,):   current colour of every vertex
#   conflicting  int64 array:        vertices that currently have >= 1 neighbour of their own colour
#   tabu_until   int64 array (n, k): (v, c) is tabu while tabu_until[v, c] > it
#   it           current iteration (starts at 1)
#   conflicts    current number of conflicting edges
#   best_conf    fewest conflicting edges seen so far in this search
#   k            number of colours available (0 .. k-1)
#   rng          random.Random, the ONLY source of randomness allowed (results must be reproducible)
# Must return (v, c): a vertex and a colour different from its current one.
#
# ===--------------------------------------------------------------------------------------===#

import numpy as np


# EVOLVE-BLOCK-START
# Global state for tracking search dynamics
_last_improvement_iter = 0
_stagnation_counter = 0

def select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng):
    """Simple but effective move selection with basic diversification."""
    global _last_improvement_iter, _stagnation_counter
    
    # Reset state at beginning of search
    if it == 1:
        _last_improvement_iter = 0
        _stagnation_counter = 0
    
    # Calculate conflict changes for all possible moves
    own = gamma[conflicting, col[conflicting]]                # conflicts each candidate causes now
    delta = gamma[conflicting] - own[:, None]                 # change in conflicts for each (v, c)
    
    # Apply aspiration criterion
    allowed = (tabu_until[conflicting] <= it) | ((conflicts + delta) < best_conf)
    
    # Create mask for valid moves (excluding current color)
    masked = np.where(allowed, delta, 1 << 30)
    masked[np.arange(conflicting.size), col[conflicting]] = 1 << 30   # keeping the colour is not a move
    
    # Track stagnation
    if conflicts < best_conf:
        _last_improvement_iter = it
        _stagnation_counter = 0
    else:
        _stagnation_counter += 1
    
    # Basic greedy selection
    flat = int(np.argmin(masked))
    i, c = divmod(flat, k)
    
    # If everything is tabu, random move
    if masked[i, c] >= (1 << 30):
        i = rng.randrange(conflicting.size)
        c = rng.randrange(k - 1)
        if c >= col[conflicting[i]]:
            c += 1
        return int(conflicting[i]), int(c)
    
    # Occasionally diversify when stagnating (simple but effective)
    if _stagnation_counter > 500 and rng.random() < 0.1:
        # Choose a move from top 30% of moves (not just the best)
        flat_indices = np.argsort(masked.flatten())
        top_moves_count = max(1, int(len(flat_indices) * 0.3))
        top_moves = flat_indices[:top_moves_count]
        
        if len(top_moves) > 0:
            chosen_flat = rng.choice(top_moves)
            i, c = divmod(chosen_flat, k)
            if masked[i, c] < (1 << 30):
                return int(conflicting[i]), int(c)
    
    return int(conflicting[i]), int(c)


def tenure(n, k, num_conflicting, it, rng):
    """Simple and robust tenure calculation."""
    # Original rule: 0.6 * num_conflicting + rand(10)
    # But with better bounds
    base_tenure = int(0.6 * num_conflicting) + rng.randrange(10)
    return max(5, min(100, base_tenure))
# EVOLVE-BLOCK-END
