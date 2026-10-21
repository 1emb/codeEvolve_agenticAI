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
    """Classic TabuCol move: among conflicting vertices, the non-tabu recolouring that
    reduces the conflict count the most (ties: first found). A tabu move is allowed if it
    would beat the best conflict count seen (aspiration). If every move is tabu, random."""
    own = gamma[conflicting, col[conflicting]]                # conflicts each candidate causes now
    delta = gamma[conflicting] - own[:, None]                 # change in conflicts for each (v, c)
    allowed = (tabu_until[conflicting] <= it) | ((conflicts + delta) < best_conf)
    masked = np.where(allowed, delta, 1 << 30)
    masked[np.arange(conflicting.size), col[conflicting]] = 1 << 30   # keeping the colour is not a move
    flat = int(np.argmin(masked))
    i, c = divmod(flat, k)
    if masked[i, c] >= (1 << 30):                             # everything tabu: random move
        i = rng.randrange(conflicting.size)
        c = rng.randrange(k - 1)
        if c >= col[conflicting[i]]:
            c += 1
    return int(conflicting[i]), int(c)


def tenure(n, k, num_conflicting, it, rng):
    """How many iterations the reverted move stays forbidden."""
    return int(0.6 * num_conflicting) + rng.randrange(10)
# EVOLVE-BLOCK-END
