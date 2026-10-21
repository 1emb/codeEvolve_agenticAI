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
_tenure_history = []
_conflict_history = []

def select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng):
    """Refined selection strategy with better exploitation of conflict information."""
    global _last_improvement_iter, _tenure_history, _conflict_history
    
    # Reset state at beginning of search
    if it == 1:
        _last_improvement_iter = 0
        _tenure_history = []
        _conflict_history = []
    
    # Track conflicts for stagnation detection
    _conflict_history.append(conflicts)
    if len(_conflict_history) > 20:
        _conflict_history.pop(0)
    
    # Calculate conflict changes for all possible moves
    own = gamma[conflicting, col[conflicting]]                # conflicts each candidate causes now
    delta = gamma[conflicting] - own[:, None]                 # change in conflicts for each (v, c)
    
    # Create mask for allowed moves (non-tabu OR aspiration)
    allowed = (tabu_until[conflicting] <= it) | ((conflicts + delta) < best_conf)
    
    # Set invalid moves to very high cost
    masked = np.where(allowed, delta, 1 << 30)
    masked[np.arange(conflicting.size), col[conflicting]] = 1 << 30   # keeping the colour is not a move
    
    # If everything is tabu, force a move with careful selection
    if np.all(masked >= (1 << 30)):
        # Try to find a non-tabu move first
        non_tabu_mask = (tabu_until[conflicting] <= it).all(axis=1)
        if non_tabu_mask.any():
            # Pick a random non-tabu vertex
            valid_vertices = conflicting[non_tabu_mask]
            i = rng.choice(valid_vertices)
            # Pick a random valid color that's not current
            c = rng.randrange(k - 1)
            if c >= col[i]:
                c += 1
            return int(i), int(c)
        else:
            # Everything is tabu - use a more strategic approach
            # Select vertex with highest conflict count (most problematic)
            conflict_counts = gamma[conflicting, col[conflicting]]
            if len(conflict_counts) > 0:
                max_conflict_idx = np.argmax(conflict_counts)
                i = conflicting[max_conflict_idx]
                # Choose color that minimizes conflicts among valid options
                vertex_deltas = delta[max_conflict_idx]
                # Prefer moves that reduce conflicts (or at least don't increase)
                valid_colors = np.where(vertex_deltas < 1)[0]  # Colors that don't increase conflicts
                if len(valid_colors) == 0:
                    # No improvement possible, pick any valid color
                    valid_colors = np.arange(k)
                valid_colors = valid_colors[valid_colors != col[i]]  # Remove current color
                
                if len(valid_colors) > 0:
                    # Pick the color that gives minimum conflict change
                    best_color_idx = np.argmin(vertex_deltas[valid_colors])
                    c = valid_colors[best_color_idx]
                else:
                    # Fallback to completely random
                    c = rng.randrange(k - 1)
                    if c >= col[i]:
                        c += 1
                return int(i), int(c)
            else:
                # Fallback to random
                i = rng.choice(conflicting)
                c = rng.randrange(k - 1)
                if c >= col[i]:
                    c += 1
                return int(i), int(c)
    
    # Find the best moves (minimum delta)
    min_delta = np.min(masked)
    
    # Get indices of all moves with minimum delta
    best_move_indices = np.where(masked == min_delta)
    
    # For better exploitation, when there are improving moves, prioritize the most beneficial
    if min_delta < 0 and len(best_move_indices[0]) > 1:
        # Among improving moves, prefer the ones that reduce conflicts most
        # Sort by ascending delta (most improving first)
        indices = list(zip(best_move_indices[0], best_move_indices[1]))
        indices.sort(key=lambda x: masked[x[0], x[1]])
        # Pick the most beneficial improvement
        i, c = indices[0]
    else:
        # Simple selection from best moves
        # If there's a single best move, take it; otherwise random selection
        if len(best_move_indices[0]) == 1:
            i = best_move_indices[0][0]
            c = best_move_indices[1][0]
        else:
            # Random selection among best moves to avoid cycling
            random_idx = rng.randrange(len(best_move_indices[0]))
            i = best_move_indices[0][random_idx]
            c = best_move_indices[1][random_idx]
    
    # Ensure the color is different from current
    if c == col[conflicting[i]]:
        # Find a different valid color
        valid_colors = [cc for cc in range(k) if cc != col[conflicting[i]]]
        if valid_colors:
            c = rng.choice(valid_colors)
    
    # Update improvement tracking
    if conflicts + masked[i, c] < conflicts:
        _last_improvement_iter = it
    
    return int(conflicting[i]), int(c)


def tenure(n, k, num_conflicting, it, rng):
    """Adaptive tenure based on recent performance and conflict levels."""
    global _tenure_history, _last_improvement_iter
    
    # Base tenure calculation
    base_tenure = max(5, int(0.6 * num_conflicting) + rng.randrange(10))
    
    # Track tenure history for adaptation
    _tenure_history.append(base_tenure)
    if len(_tenure_history) > 10:
        _tenure_history.pop(0)
    
    # Adaptive component based on recent performance
    if len(_tenure_history) >= 3:
        recent_tenure = _tenure_history[-3:]
        avg_tenure = np.mean(recent_tenure)
        
        # If we haven't improved in a while, increase tenure to encourage exploration
        if it - _last_improvement_iter > 2000:
            base_tenure = int(avg_tenure * 1.2)
        elif it - _last_improvement_iter < 500:
            # If we're making good progress, decrease tenure for faster convergence
            base_tenure = int(avg_tenure * 0.8)
    
    # Add some randomness but keep it bounded
    random_factor = rng.randint(-2, 3)
    adjusted_tenure = max(3, base_tenure + random_factor)
    
    # Clamp to reasonable bounds
    return min(max(adjusted_tenure, 5), 100)
# EVOLVE-BLOCK-END
