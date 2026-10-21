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


# HAND-EDITED COPY of island_6/best_sol.py (the evolved artifact). One change, marked (a) below:
#   (a) softmax sampling draws from the supplied rng instead of numpy.random (reproducibility).
# EVOLVE-BLOCK-START
import numpy as np

# Global state for tracking search dynamics
_stagnation_count = 0
_last_conflicts = None
_last_improvement_iter = 0
_tenure_history = []
_exploration_phase = True

def select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng):
    """Enhanced selection strategy with sophisticated exploitation/exploration balance."""
    global _stagnation_count, _last_conflicts, _last_improvement_iter, _exploration_phase
    
    # Reset at start of each run
    if it == 1:
        _stagnation_count = 0
        _last_conflicts = None
        _last_improvement_iter = 0
        _tenure_history = []
        _exploration_phase = True
    
    if len(conflicting) == 0:
        return 0, 0
    
    # Calculate conflict reduction potential for each move
    own = gamma[conflicting, col[conflicting]]                # conflicts each candidate causes now
    delta = gamma[conflicting] - own[:, None]                 # change in conflicts for each (v, c)
    
    # Apply aspiration criterion - if improving move, ignore tabu status
    allowed = (tabu_until[conflicting] <= it) | ((conflicts + delta) < best_conf)
    
    # Mask forbidden moves and prevent staying on current color
    masked = np.where(allowed, delta, np.inf)
    masked[np.arange(conflicting.size), col[conflicting]] = np.inf
    
    # Track stagnation
    if _last_conflicts is not None:
        if _last_conflicts == conflicts:
            _stagnation_count += 1
        else:
            _stagnation_count = 0
            _last_improvement_iter = it
    _last_conflicts = conflicts
    
    # Determine exploration vs exploitation phase
    if it > 50000 and _stagnation_count > 5:
        _exploration_phase = True
    elif it > 20000 and conflicts < _last_conflicts and _stagnation_count < 3:
        _exploration_phase = False
    
    # Use greedy selection when there are improving moves or in early phases
    min_delta = np.min(masked)
    
    if min_delta < 0 or it < 20000 or _exploration_phase == False:
        # Greedy selection: pick the best improving move
        flat = int(np.argmin(masked))
        i, c = divmod(flat, k)
        return int(conflicting[i]), int(c)
    else:
        # Stochastic selection with better exploration control
        # When no improving moves, we want to explore to escape local minima
        
        # Increase randomness when stagnating or in early iterations
        if _stagnation_count > 3 or it < 30000:
            # Use epsilon-greedy approach with dynamic exploration rate
            if _stagnation_count > 5:
                epsilon = 0.4
            elif it < 50000:
                epsilon = 0.3
            else:
                epsilon = 0.2
            
            # With probability epsilon, select randomly
            if rng.random() < epsilon:
                # Select random valid move
                i = rng.randrange(len(conflicting))
                v = conflicting[i]
                valid_colors = [j for j in range(k) if j != col[v]]
                c = valid_colors[rng.randrange(len(valid_colors))]
                return int(v), int(c)
        
        # Otherwise, use softmax selection for diversity with better numerical stability
        try:
            # Calculate utilities (negative deltas since lower delta is better)
            utilities = -delta
            
            # Clip utilities to prevent numerical overflow
            utilities = np.clip(utilities, -100, 100)
            
            # Apply softmax with temperature that decreases over time
            temp = max(0.1, 0.5 * (1.0 - it / 200000.0))
            
            # Shift utilities for numerical stability
            shifted_utilities = utilities - np.max(utilities[utilities < np.inf])
            
            # Compute exponentials
            exp_utilities = np.exp(shifted_utilities / temp)
            
            # Set invalid moves to 0 probability
            valid_mask = (masked < np.inf)
            exp_utilities[~valid_mask] = 0
            
            # Normalize probabilities
            total_prob = np.sum(exp_utilities)
            if total_prob > 0:
                probabilities = exp_utilities / total_prob
            else:
                # Fallback to uniform distribution among valid moves
                valid_indices = np.where(valid_mask)[0]
                probabilities = np.zeros_like(exp_utilities)
                if len(valid_indices) > 0:
                    probabilities[valid_indices] = 1.0 / len(valid_indices)
            
            # Sample move according to probabilities
            flat_probs = probabilities.flatten()
            flat_indices = np.arange(len(flat_probs))
            cum = np.cumsum(flat_probs)
            if cum[-1] <= 0:
                raise ValueError("no valid move")          # -> random fallback below, as before
            sampled_flat = min(int(np.searchsorted(cum, rng.random() * cum[-1], side="right")), len(flat_probs) - 1)
            
            i, c = divmod(sampled_flat, k)
            return int(conflicting[i]), int(c)
        except Exception:
            # Fallback to simple random selection if anything goes wrong
            i = rng.randrange(len(conflicting))
            v = conflicting[i]
            valid_colors = [j for j in range(k) if j != col[v]]
            c = valid_colors[rng.randrange(len(valid_colors))]
            return int(v), int(c)


def tenure(n, k, num_conflicting, it, rng):
    """Adaptive tenure with sophisticated stagnation detection and response."""
    global _stagnation_count, _last_conflicts, _last_improvement_iter, _tenure_history
    
    # Reset at start
    if it == 1:
        _stagnation_count = 0
        _last_conflicts = None
        _last_improvement_iter = 0
        _tenure_history = []
    
    # Base tenure from classic formula with some improvements
    base_tenure = max(3, int(0.5 * num_conflicting) + rng.randrange(8))
    
    # Track recent progress for stagnation detection
    _tenure_history.append(num_conflicting)
    if len(_tenure_history) > 50:
        _tenure_history.pop(0)
    
    # Detect stagnation more accurately with recent window
    if len(_tenure_history) >= 10:
        # Check if there's been significant improvement recently
        recent_improvement = False
        recent_worsening = False
        
        # Look at last 20 iterations for improvement
        if len(_tenure_history) >= 20:
            for i in range(1, min(20, len(_tenure_history))):
                if _tenure_history[-1] < _tenure_history[-1-i]:
                    recent_improvement = True
                    break
                elif _tenure_history[-1] > _tenure_history[-1-i]:
                    recent_worsening = True
                    break
        
        # Update stagnation count based on recent progress
        if recent_improvement:
            _stagnation_count = max(0, _stagnation_count - 1)
        elif recent_worsening:
            _stagnation_count += 1
        else:
            # No clear trend - keep current stagnation level
            pass
    
    # Adjust tenure based on stagnation and progress
    if _stagnation_count > 5:
        # Increase tenure significantly when stagnating
        base_tenure = int(base_tenure * 1.8)
    elif _stagnation_count > 3:
        # Moderate increase for moderate stagnation
        base_tenure = int(base_tenure * 1.5)
    elif it > 10000 and _stagnation_count == 0:
        # Decrease tenure when making progress in later stages
        base_tenure = max(3, int(base_tenure * 0.8))
    elif it > 10000 and _stagnation_count > 0:
        # Slight increase when making progress but still stagnating
        base_tenure = int(base_tenure * 1.2)
    
    # Ensure reasonable bounds
    final_tenure = max(3, min(100, base_tenure))
    
    _last_conflicts = num_conflicting
    
    return final_tenure
# EVOLVE-BLOCK-END
