#!/usr/bin/env python
# Held-out test for an evolved hook program: run the FIXED decreasing-k wrapper with the program's
# select_move/tenure under a wall-clock limit on a list of graphs and report colours used.
#   python heldout_hooks.py <hooks.py> [<hooks2.py> ...] --graphs DSJC125.5,DSJC1000.5,queen8_8,myciel5 --time-limit 10
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[_v] = "1"
import argparse, importlib.util, sys, time
from concurrent.futures import ProcessPoolExecutor
COMMON = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, COMMON)
import tabucol_framework as fw
from scoring import read_dimacs
from make_problem import BEST_KNOWN
DATA = os.path.expanduser("~/codeEvolve/data/dimacs")

def load_hooks(path):
    spec = importlib.util.spec_from_file_location("hooks_" + str(abs(hash(path))), path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def one(args):
    path, g, tl = args
    hooks = load_hooks(path); n, edges = read_dimacs(os.path.join(DATA, f"{g}.col")); edges0 = [(u-1, v-1) for u, v in edges]
    t0 = time.perf_counter(); col = fw.color_graph(n, edges0, tl, hooks.select_move, hooks.tenure); t = time.perf_counter() - t0
    bad = sum(1 for u, v in edges0 if col[u] == col[v])
    return path, g, (len(set(col)) if bad == 0 else f"INVALID({bad})"), round(t, 1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("programs", nargs="+"); ap.add_argument("--graphs", required=True); ap.add_argument("--time-limit", type=float, default=10.0)
    a = ap.parse_args(); graphs = a.graphs.split(",")
    jobs = [(p, g, a.time_limit) for p in a.programs for g in graphs]
    with ProcessPoolExecutor(min(32, len(jobs))) as ex: res = list(ex.map(one, jobs))
    names = [os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p) for p in a.programs]
    print(f"time limit {a.time_limit:g} s per graph; colours used (best known in parentheses)")
    print(f"{'graph':12}" + "".join(f"{nm[-28:]:>30}" for nm in names))
    for g in graphs:
        row = f"{g:12}"
        for p in a.programs:
            k = [r[2] for r in res if r[0] == p and r[1] == g][0]; row += f"{str(k) + ' (' + str(BEST_KNOWN[g]) + ')':>30}"
        print(row)

if __name__ == "__main__": main()
