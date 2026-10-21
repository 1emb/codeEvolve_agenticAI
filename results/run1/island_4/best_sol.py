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

def select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng):
    """Simplified yet effective TabuCol move selection"""
    global stagnation_counter, last_improvement_iter
    
    # Reset state at beginning of each run
    if it == 1:
        stagnation_counter = 0
        last_improvement_iter = 0
    
    # Calculate conflict changes for all possible moves
    own = gamma[conflicting, col[conflicting]]
    delta = gamma[conflicting] - own[:, None]
    
    # Apply aspiration criterion: allow moves that improve best_conf
    allowed = (tabu_until[conflicting] <= it) | ((conflicts + delta) < best_conf)
    
    # Mask invalid moves (including keeping current color)
    masked = np.where(allowed, delta, 1 << 30)
    masked[np.arange(conflicting.size), col[conflicting]] = 1 << 30
    
    # Simple but effective: prefer moves that reduce conflicts the most
    # With some randomness to escape local minima
    if len(conflicting) > 0:
        # Find the minimum value (best improvement)
        min_delta = np.min(masked)
        
        # If there's a clear best move, prefer it with some probability
        # Otherwise, make a more stochastic choice
        if min_delta < 1000000:  # Some reasonable threshold
            # Get indices where we achieve the minimum
            best_indices = np.where(masked == min_delta)
            
            # If we have multiple equally good moves, randomly choose one
            if len(best_indices[0]) > 1:
                # Use simple random selection among best moves
                idx = rng.randrange(len(best_indices[0]))
                i = best_indices[0][idx]
                c = best_indices[1][idx]
            else:
                # Only one best move
                i = best_indices[0][0]
                c = best_indices[1][0]
        else:
            # No good moves found - fall back to standard greedy
            flat = int(np.argmin(masked))
            i, c = divmod(flat, k)
            
            # If everything is tabu, choose a random valid move
            if masked[i, c] >= (1 << 30):
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
                    # All moves are tabu - pick random move
                    i = rng.randrange(conflicting.size)
                    c = rng.randrange(k - 1)
                    if c >= col[conflicting[i]]:
                        c += 1
    else:
        # No conflicting vertices, pick a random vertex and color
        i = rng.randrange(len(conflicting))
        c = rng.randrange(k - 1)
        if c >= col[conflicting[i]]:
            c += 1
    
    return int(conflicting[i]), int(c)


def tenure(n, k, num_conflicting, it, rng):
    """Refined adaptive tenure with better stabilization"""
    global stagnation_counter, last_improvement_iter
    
    # Base tenure calculation with reduced variance
    base_tenure = max(5, int(0.5 * num_conflicting) + rng.randrange(12))
    
    # Adjust based on stagnation - more conservative than before
    if it - last_improvement_iter > 4000:
        stagnation_counter += 1
        # Increase tenure when stagnating to encourage diversification
        # But be more conservative about increases
        multiplier = 1.0 + 0.05 * min(stagnation_counter, 10)
        base_tenure = max(base_tenure, int(base_tenure * multiplier))
    else:
        stagnation_counter = 0
        last_improvement_iter = it
    
    # Ensure reasonable bounds
    return max(5, min(base_tenure, 100))
# EVOLVE-BLOCK-END
