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
def select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng):
    """
    Simplified but effective TabuCol move selection inspired by successful approaches.
    Focuses on robust tie-breaking and reliable aspiration criteria.
    """
    # Calculate conflict reduction potential for all moves
    own = gamma[conflicting, col[conflicting]]                # conflicts each candidate causes now
    delta = gamma[conflicting] - own[:, None]                 # change in conflicts for each (v, c)
    
    # Apply aspiration criterion: allow moves that reduce conflicts below best seen
    # or moves that aren't tabu
    allowed = (tabu_until[conflicting] <= it) | ((conflicts + delta) < best_conf)
    
    # Create mask for valid moves (excluding current color)
    valid_mask = np.ones_like(delta, dtype=bool)
    valid_mask[np.arange(conflicting.size), col[conflicting]] = False
    
    # Combine allowed and valid conditions
    final_mask = allowed & valid_mask
    
    # For moves that are not allowed, set their delta to a large positive value
    masked_delta = np.where(final_mask, delta, 1 << 30)
    
    # If all moves are forbidden, pick a random valid move
    if np.all(masked_delta >= (1 << 30)):
        # Fallback: pick a random conflicting vertex and random color
        v_idx = rng.randrange(len(conflicting))
        v = conflicting[v_idx]
        c = rng.randrange(k - 1)
        if c >= col[v]:
            c += 1
        return int(v), int(c)
    
    # Find all moves with minimum delta (best moves)
    min_delta = np.min(masked_delta)
    
    # Get indices of all best moves
    best_move_indices = np.where(masked_delta == min_delta)
    
    # If there's more than one best move, randomly break ties
    # This prevents cycling and adds useful diversification
    if len(best_move_indices[0]) > 1:
        idx = rng.randrange(len(best_move_indices[0]))
        v_idx = best_move_indices[0][idx]
        c_idx = best_move_indices[1][idx]
    else:
        # Only one best move, use it directly
        v_idx = best_move_indices[0][0]
        c_idx = best_move_indices[1][0]
    
    return int(conflicting[v_idx]), int(c_idx)


def tenure(n, k, num_conflicting, it, rng):
    """
    Simple but effective tenure calculation based on proven principles.
    """
    # Base tenure - classic formula with small adjustments
    base_tenure = max(5, int(0.6 * num_conflicting) + rng.randrange(10))
    
    # Ensure reasonable bounds
    return max(3, min(50, base_tenure))
# EVOLVE-BLOCK-END
