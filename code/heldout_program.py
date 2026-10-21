#!/usr/bin/env python
# Held-out test for whole colouring programs (color_graph(n, edges, time_limit)): run each program on each
# graph for a wall-clock limit, over several evaluation seeds (seed 0 = graph as given, others = isomorphic
# copies with relabelled vertices and shuffled edges, exactly as evaluate_multi.py does), verify the colouring
# independently and report colours used per seed.
#   python heldout_program.py <prog.py> [<prog2.py> ...] --graphs DSJC125.5,DSJC1000.5,queen8_8,myciel5 --time-limit 10 --seeds 3
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[_v] = "1"
import argparse, importlib.util, random, signal, sys, time
from concurrent.futures import ProcessPoolExecutor
COMMON = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, COMMON)
from scoring import read_dimacs
from make_problem import BEST_KNOWN
DATA = os.path.expanduser("~/codeEvolve/data/dimacs")

class Timeout(Exception): pass
def _alarm(s, f): raise Timeout()

def load(path):
    spec = importlib.util.spec_from_file_location("prog_" + str(abs(hash(path))), path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def one(args):
    path, g, tl, seed = args
    n, edges = read_dimacs(os.path.join(DATA, f"{g}.col")); edges0 = [(u - 1, v - 1) for u, v in edges]
    if seed != 0:
        prm = random.Random(seed); perm = list(range(n)); prm.shuffle(perm)
        edges0 = [(perm[u], perm[v]) for u, v in edges0]; prm.shuffle(edges0)
    prog = load(path); random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception: pass
    signal.signal(signal.SIGALRM, _alarm); signal.setitimer(signal.ITIMER_REAL, tl); t0 = time.perf_counter()
    try:
        col = list(prog.color_graph(n, edges0, tl)); t = time.perf_counter() - t0
    except Timeout:
        return path, g, seed, "TIMEOUT", tl
    except Exception as e:
        return path, g, seed, f"ERROR({type(e).__name__})", time.perf_counter() - t0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if len(col) != n: return path, g, seed, f"BAD_LEN({len(col)})", t
    bad = sum(1 for u, v in edges0 if col[u] == col[v])
    return path, g, seed, (len(set(col)) if bad == 0 else f"INVALID({bad})"), round(t, 1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("programs", nargs="+"); ap.add_argument("--graphs", required=True)
    ap.add_argument("--time-limit", type=float, default=10.0); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(); graphs = a.graphs.split(","); seeds = list(range(a.seeds))
    jobs = [(p, g, a.time_limit, s) for p in a.programs for g in graphs for s in seeds]
    with ProcessPoolExecutor(min(a.workers, len(jobs))) as ex: res = list(ex.map(one, jobs))
    names = [os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p) for p in a.programs]
    print(f"time limit {a.time_limit:g} s per graph, seeds {seeds}; colours used per seed (best known in parentheses); every count is a verified conflict-free colouring")
    print(f"{'graph':12}" + "".join(f"{nm[-30:]:>34}" for nm in names))
    for g in graphs:
        row = f"{g:12}"
        for p in a.programs:
            ks = [r[3] for r in res if r[0] == p and r[1] == g]
            row += f"{'/'.join(str(k) for k in ks) + ' (' + str(BEST_KNOWN[g]) + ')':>34}"
        print(row)

if __name__ == "__main__": main()
