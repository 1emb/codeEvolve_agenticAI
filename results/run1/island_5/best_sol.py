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
import numpy as np

# Global state for tracking search behavior
stagnation_counter = 0
last_improvement_iter = 0
last_conflicts = None

def select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng):
    """Simplified and robust TabuCol move selection"""
    global stagnation_counter, last_improvement_iter, last_conflicts
    
    # Reset state at beginning of each run
    if it == 1:
        stagnation_counter = 0
        last_improvement_iter = 0
        last_conflicts = None
    
    # Calculate conflict changes for all possible moves
    own = gamma[conflicting, col[conflicting]]
    delta = gamma[conflicting] - own[:, None]
    
    # Apply aspiration criterion: allow moves that improve best_conf
    allowed = (tabu_until[conflicting] <= it) | ((conflicts + delta) < best_conf)
    
    # Mask invalid moves (including keeping current color)
    LARGE_NUMBER = 1 << 30
    masked = np.where(allowed, delta, LARGE_NUMBER)
    masked[np.arange(conflicting.size), col[conflicting]] = LARGE_NUMBER
    
    # Track stagnation
    if last_conflicts is not None:
        if last_conflicts == conflicts:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            last_improvement_iter = it
    last_conflicts = conflicts
    
    # Simplified greedy selection - always prefer moves that reduce conflicts
    # When we have a clear choice, use it. Otherwise, be more conservative.
    flat = int(np.argmin(masked))
    i, c = divmod(flat, k)
    
    # If everything is tabu, choose a random non-tabu move
    if masked[i, c] >= LARGE_NUMBER:
        # Try to find a non-tabu move
        non_tabu_mask = tabu_until[conflicting] <= it
        if np.any(non_tabu_mask):
            # Find a random valid move
            valid_indices = np.where(non_tabu_mask)[0]
            i = rng.choice(valid_indices)
            # Find a valid color
            valid_colors = [color for color in range(k) if color != col[conflicting[i]]]
            if valid_colors:
                c = rng.choice(valid_colors)
            else:
                # Should not happen, but just in case
                c = (col[conflicting[i]] + 1) % k
        else:
            # All moves are tabu - pick random move but ensure it's valid
            i = rng.randrange(conflicting.size)
            # Find a valid color different from current
            current_color = col[conflicting[i]]
            valid_colors = list(range(current_color)) + list(range(current_color + 1, k))
            if valid_colors:
                c = rng.choice(valid_colors)
            else:
                # Fallback - shouldn't happen with valid inputs
                c = (current_color + 1) % k
    
    return int(conflicting[i]), int(c)


def tenure(n, k, num_conflicting, it, rng):
    """Simple and effective tenure calculation"""
    global stagnation_counter, last_improvement_iter, last_conflicts
    
    # Reset state at beginning of each run
    if it == 1:
        last_conflicts = None
    
    # Simple tenure based on number of conflicting vertices
    # Slightly higher than before to allow more exploration
    base_tenure = int(0.6 * num_conflicting) + rng.randrange(10)
    
    # Adjust based on recent progress
    if last_conflicts is not None:
        if last_conflicts > num_conflicting:
            # Improvement occurred - reduce tenure to be more aggressive
            base_tenure = max(2, int(base_tenure * 0.7))
        elif last_conflicts == num_conflicting:
            # No improvement - increase tenure slightly
            base_tenure = int(base_tenure * 1.1)
        else:
            # Getting worse - increase tenure significantly
            base_tenure = int(base_tenure * 1.3)
    
    # Additional adjustment for stagnation
    if stagnation_counter > 3:
        base_tenure = int(base_tenure * 1.5)
    
    # Ensure reasonable bounds
    final_tenure = max(3, min(150, base_tenure))
    
    last_conflicts = num_conflicting
    
    return final_tenure
# EVOLVE-BLOCK-END
