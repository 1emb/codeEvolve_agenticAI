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
_stagnation_count = 0
_tenure_history = []
_last_conflicts = 0

def select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng):
    global _stagnation_count, _last_conflicts
    
    # Reset at start of each run
    if it == 1:
        _stagnation_count = 0
        _tenure_history = []
        _last_conflicts = conflicts
    
    # Calculate conflict reduction potential for each move
    own = gamma[conflicting, col[conflicting]]                # conflicts each candidate causes now
    delta = gamma[conflicting] - own[:, None]                 # change in conflicts for each (v, c)
    
    # Apply aspiration criterion - if improving move, ignore tabu status
    allowed = (tabu_until[conflicting] <= it) | ((conflicts + delta) < best_conf)
    
    # Mask forbidden moves and prevent staying on current color
    masked = np.where(allowed, delta, 1 << 30)
    masked[np.arange(conflicting.size), col[conflicting]] = 1 << 30
    
    # Check if there's at least one improving move
    min_delta = np.min(masked)
    
    # Use greedy selection when there are improving moves or early in search
    if min_delta < 0 or it < 5000:
        # Greedy selection: pick the best improving move
        flat = int(np.argmin(masked))
        i, c = divmod(flat, k)
        return int(conflicting[i]), int(c)
    else:
        # Stochastic selection with exploration bias
        # When no improving moves, we want to explore to escape local minima
        
        # Increase randomness when stagnating or in early iterations
        if it < 5000 or _stagnation_count > 3:
            # Use epsilon-greedy approach with exploration
            epsilon = 0.2 if _stagnation_count > 3 else 0.1
            
            # With probability epsilon, select randomly
            if rng.random() < epsilon:
                # Select random valid move
                i = rng.randrange(len(conflicting))
                v = conflicting[i]
                valid_colors = [j for j in range(k) if j != col[v]]
                c = valid_colors[rng.randrange(len(valid_colors))]
                return int(v), int(c)
        
        # Otherwise, use softmax selection for diversity
        scores = np.where(allowed, delta, float('inf'))
        scores = scores.astype(np.float32)
        
        # Add small noise to encourage exploration
        noise = np.random.rand(*scores.shape) * 0.1
        scores = scores + noise
        
        # Ensure we don't select current color
        scores[np.arange(conflicting.size), col[conflicting]] = float('inf')
        
        # Select best valid move
        flat = int(np.argmin(scores))
        i, c = divmod(flat, k)
        
        # Safety check
        if c == col[int(conflicting[i])]:
            # Find a different valid color
            valid_colors = [j for j in range(k) if j != col[int(conflicting[i])]]
            c = valid_colors[rng.randrange(len(valid_colors))]
            
        return int(conflicting[i]), int(c)


def tenure(n, k, num_conflicting, it, rng):
    global _stagnation_count, _tenure_history, _last_conflicts
    
    # Reset at start
    if it == 1:
        _stagnation_count = 0
        _tenure_history = []
        _last_conflicts = num_conflicting
    
    # Track recent progress for stagnation detection
    _tenure_history.append(num_conflicting)
    if len(_tenure_history) > 20:
        _tenure_history.pop(0)
    
    # Detect stagnation: if no improvement in last 50 iterations
    if len(_tenure_history) >= 20:
        recent_improvement = False
        for i in range(1, min(50, len(_tenure_history))):
            if _tenure_history[-1] < _tenure_history[-1-i]:
                recent_improvement = True
                break
        if not recent_improvement:
            _stagnation_count += 1
        else:
            _stagnation_count = 0
    
    # Base tenure from classic formula
    base_tenure = int(0.6 * num_conflicting) + rng.randrange(10)
    
    # Adjust tenure based on stagnation and progress
    if _stagnation_count > 5:
        # Increase tenure significantly when stagnating
        base_tenure = int(base_tenure * 1.5)
    elif it > 10000 and _stagnation_count > 2:
        # Moderate increase for moderate stagnation
        base_tenure = int(base_tenure * 1.2)
    elif it > 10000 and _stagnation_count == 0:
        # Decrease tenure when making progress in later stages
        base_tenure = max(5, int(base_tenure * 0.9))
    
    # Ensure reasonable bounds
    return max(5, min(base_tenure, 100))
# EVOLVE-BLOCK-END
