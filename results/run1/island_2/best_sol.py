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
# Global state for adaptive mechanisms
_stagnation_counter = 0
_last_improvement_it = 0

def select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng):
    global _stagnation_counter, _last_improvement_it
    
    # Reset state at beginning of each run
    if it == 1:
        _stagnation_counter = 0
        _last_improvement_it = 0
    
    # Track stagnation
    if conflicts < best_conf:
        _last_improvement_it = it
        _stagnation_counter = 0
    else:
        _stagnation_counter += 1
    
    # Calculate conflict changes for all valid moves
    own = gamma[conflicting, col[conflicting]]
    delta = gamma[conflicting] - own[:, None]
    
    # Apply aspiration criterion: allow moves that improve best solution
    allowed = (tabu_until[conflicting] <= it) | ((conflicts + delta) < best_conf)
    
    # Create a scoring matrix with penalties for tabu moves
    masked = np.where(allowed, delta, 1 << 30)
    masked[np.arange(conflicting.size), col[conflicting]] = 1 << 30
    
    # If all moves are tabu, apply fallback strategy
    if np.all(masked >= (1 << 30)):
        # Fallback: pick vertex based on conflict count (more conflicts = higher priority)
        conflict_counts = gamma[conflicting, col[conflicting]]
        # Weight by conflict count (higher weight to more conflicted vertices)
        weights = np.maximum(1, conflict_counts)
        weights = weights / np.sum(weights)
        
        # Choose vertex based on weights
        selected_idx = np.random.choice(len(conflicting), p=weights) if len(conflicting) > 0 else 0
        i = selected_idx
        
        # Choose color that minimizes conflicts with neighbors
        v = conflicting[i]
        neighbor_conflicts = gamma[v]
        # Prefer colors that appear less frequently among neighbors
        color_scores = neighbor_conflicts.copy()
        color_scores[col[v]] = 1000  # Penalize current color strongly
        # Add small random noise to break ties
        color_scores += np.random.random(k) * 0.01
        c = int(np.argmin(color_scores))
        
        return int(v), int(c)
    
    # Select the best move (minimum conflict reduction)
    flat = int(np.argmin(masked))
    i, c = divmod(flat, k)
    
    # Enhanced escape mechanism for better performance
    # When we're stagnating, sometimes take a move that reduces conflicts by a small amount
    if _stagnation_counter > 500 and rng.random() < 0.03:
        # Find all valid moves (not tabu and not current color)
        valid_mask = (tabu_until[conflicting] <= it) & (np.arange(k) != col[conflicting][:, None])
        valid_indices = np.where(valid_mask)
        if len(valid_indices[0]) > 0:
            # Get all valid moves and their delta values
            valid_deltas = delta[valid_indices]
            # Select among top 30% of moves (or all if fewer)
            top_count = max(1, len(valid_deltas) // 3)
            if top_count < len(valid_deltas):
                # Sort by delta (conflict reduction) and take top moves
                sorted_indices = np.argsort(valid_deltas)[:top_count]
                # Pick one of the top moves randomly
                chosen_idx = rng.choice(sorted_indices)
                # Map back to vertex and color indices
                valid_v = valid_indices[0][chosen_idx]
                valid_c = valid_indices[1][chosen_idx]
                return int(conflicting[valid_v]), int(valid_c)
    
    return int(conflicting[i]), int(c)


def tenure(n, k, num_conflicting, it, rng):
    """Adaptive tenure with better responsiveness to search dynamics."""
    # Base tenure calculation - responsive to problem size and iteration
    base_tenure = max(3, int(0.4 * num_conflicting) + 2)
    
    # Increase tenure during stagnation to encourage exploration
    if _stagnation_counter > 500:
        base_tenure = max(base_tenure, int(0.7 * num_conflicting) + 5)
    
    # Add some randomness but keep it bounded
    return max(3, base_tenure + rng.randrange(-2, 3))
# EVOLVE-BLOCK-END
