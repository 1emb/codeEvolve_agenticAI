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
# Global state for tracking search progress
_last_conflicts = None
_stagnation_count = 0

def select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng):
    """Improved move selection with better exploitation of aspiration and smarter fallback."""
    global _last_conflicts, _stagnation_count
    
    # Reset for new run
    if it == 1:
        _last_conflicts = None
        _stagnation_count = 0
    
    # Calculate conflict changes for all moves
    own = gamma[conflicting, col[conflicting]]                # conflicts each candidate causes now
    delta = gamma[conflicting] - own[:, None]                 # change in conflicts for each (v, c)
    
    # Apply aspiration criterion - allow moves that improve best solution
    allowed = (tabu_until[conflicting] <= it) | ((conflicts + delta) < best_conf)
    
    # Create penalty matrix for tabu moves
    masked = np.where(allowed, delta, 1 << 30)
    masked[np.arange(conflicting.size), col[conflicting]] = 1 << 30   # keeping the colour is not a move
    
    # Track stagnation
    if _last_conflicts is not None and conflicts == _last_conflicts:
        _stagnation_count += 1
    else:
        _stagnation_count = 0
    _last_conflicts = conflicts
    
    # When stagnating, add some randomness to escape local optima
    if _stagnation_count > 500 and rng.random() < 0.15:
        # Choose a random valid move instead of greedy
        flat = rng.randrange(masked.size)
        i, c = divmod(flat, k)
        # Ensure we're not choosing the same color
        if c == col[conflicting[i]]:
            valid_colors = [color for color in range(k) if color != col[conflicting[i]]]
            if valid_colors:
                c = rng.choice(valid_colors)
            else:
                c = rng.randrange(k - 1)
                if c >= col[conflicting[i]]:
                    c += 1
        return int(conflicting[i]), int(c)
    
    # Standard greedy selection
    flat = int(np.argmin(masked))
    i, c = divmod(flat, k)
    
    # If everything is tabu, make a reasonable choice
    if masked[i, c] >= (1 << 30):
        # Choose a random vertex from conflicting ones
        i = rng.randrange(conflicting.size)
        # Choose a random color different from current
        valid_colors = [color for color in range(k) if color != col[conflicting[i]]]
        if valid_colors:
            c = rng.choice(valid_colors)
        else:
            c = rng.randrange(k - 1)
            if c >= col[conflicting[i]]:
                c += 1
    
    return int(conflicting[i]), int(c)


def tenure(n, k, num_conflicting, it, rng):
    """Improved adaptive tenure that balances exploration and exploitation."""
    # Base tenure with some randomness
    base_tenure = int(0.6 * num_conflicting) + rng.randrange(10)
    
    # Slightly more aggressive tenure for larger problems
    if n > 400:
        base_tenure = int(base_tenure * 1.2)
    
    # Add some adaptive component based on iteration
    if it > 100000:
        # In later stages, use slightly longer tenure to avoid cycling
        base_tenure = int(base_tenure * 1.1)
    elif it > 50000:
        # Mid-stage: moderate tenure
        base_tenure = int(base_tenure * 1.05)
    
    # Ensure reasonable bounds
    return max(5, min(base_tenure, 100))
# EVOLVE-BLOCK-END
