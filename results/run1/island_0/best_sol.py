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
_stagnation_counter = 0
_last_improvement_it = 0
_tenure_history = []

def select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng):
    global _stagnation_counter, _last_improvement_it
    
    # Reset state at beginning of each run
    if it == 1:
        _stagnation_counter = 0
        _last_improvement_it = 0
    
    # Compute conflict changes for all possible moves
    own = gamma[conflicting, col[conflicting]]                # conflicts each candidate causes now
    delta = gamma[conflicting] - own[:, None]                 # change in conflicts for each (v, c)
    
    # Apply tabu restrictions and aspiration criteria
    # A move is allowed if it's not tabu OR it improves best_conf
    allowed = (tabu_until[conflicting] <= it) | ((conflicts + delta) < best_conf)
    
    # Mask tabu moves and self-moves
    masked = np.where(allowed, delta, 1 << 30)
    masked[np.arange(conflicting.size), col[conflicting]] = 1 << 30   # keeping the colour is not a move
    
    # Find the minimum value
    min_val = np.min(masked)
    
    # If all moves are tabu or equally bad, do random move
    if min_val >= (1 << 30):
        # Random selection among conflicting vertices
        i = rng.randrange(conflicting.size)
        c = rng.randrange(k - 1)
        if c >= col[conflicting[i]]:
            c += 1
        return int(conflicting[i]), int(c)
    
    # Find all moves with minimum value
    min_indices = np.where(masked == min_val)
    # Break ties randomly
    tie_indices = list(zip(min_indices[0], min_indices[1]))
    chosen_idx = rng.choice(tie_indices)
    
    # Track stagnation - if we haven't improved in a while, be more exploratory
    if conflicts < best_conf:
        _last_improvement_it = it
        _stagnation_counter = 0
    else:
        _stagnation_counter = it - _last_improvement_it
    
    # Occasionally accept a slightly worse move to escape local optima
    if _stagnation_counter > 200 and rng.random() < 0.1 and min_val > 0:
        # Accept any non-tabu move, even if it increases conflicts
        non_tabu_mask = tabu_until[conflicting] <= it
        if np.any(non_tabu_mask):
            allowed_indices = np.where(non_tabu_mask)[0]
            chosen_idx = rng.choice(allowed_indices)
            v = conflicting[chosen_idx]
            c = rng.randrange(k - 1)
            if c >= col[v]:
                c += 1
            return int(v), int(c)
    
    return int(conflicting[chosen_idx[0]]), int(chosen_idx[1])


def tenure(n, k, num_conflicting, it, rng):
    global _tenure_history
    
    # Base tenure with dynamic adjustment
    base_tenure = int(0.6 * num_conflicting) + 1
    
    # Increase tenure when we're not making progress
    if len(_tenure_history) > 10:
        recent_changes = [_tenure_history[i] - _tenure_history[i-1] 
                         for i in range(len(_tenure_history)-1, max(0, len(_tenure_history)-5), -1)]
        avg_change = sum(recent_changes) / len(recent_changes) if recent_changes else 0
        if avg_change > 0.5:  # Recent tenures have been increasing, meaning we're stuck
            base_tenure = int(base_tenure * 1.3)
    
    # Occasionally use longer tenure to help escape local optima
    if num_conflicting > 10 and rng.random() < 0.15:
        base_tenure = int(base_tenure * 1.5)
    
    # Ensure minimum tenure
    tenure_val = max(1, base_tenure) + rng.randrange(5)
    
    # Store for future use
    _tenure_history.append(tenure_val)
    
    return tenure_val
# EVOLVE-BLOCK-END
