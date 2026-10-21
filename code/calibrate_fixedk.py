#!/usr/bin/env python
# Run the starting hooks at candidate target k on each graph for a given iteration budget.
#   python calibrate_fixedk.py --graphs DSJC250.5,le450_25c,DSJC500.5 --budget 200000 [--hooks init_hooks.py]
# Prints conflicts left / solve iteration / wall time per (graph, k) and suggests a target per graph
# (the lowest k at which the start leaves between 1 and 10 conflicts).
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[_v] = "1"
import argparse, importlib.util, random, sys, time
from concurrent.futures import ProcessPoolExecutor
COMMON = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, COMMON)
import tabucol_framework as fw
from scoring import read_dimacs
from make_problem import BEST_KNOWN
DATA = os.path.expanduser("~/codeEvolve/data/dimacs")

def load_hooks(path):
    spec = importlib.util.spec_from_file_location("hooks", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def one(args):
    g, k, budget, hooks_path = args
    hooks = load_hooks(hooks_path)
    n, edges = read_dimacs(os.path.join(DATA, f"{g}.col")); edges0 = [(u-1, v-1) for u, v in edges]
    adj = fw.build_adjacency(n, edges0); rng = random.Random(0); base = fw.dsatur(n, adj)
    col = fw.start_coloring(base, k, rng); t0 = time.perf_counter()
    _, bc, sa, it = fw.fixed_k_search(n, adj, k, col, budget, rng, hooks.select_move, hooks.tenure)
    return g, k, bc, sa, it, time.perf_counter() - t0, max(base) + 1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graphs", required=True); ap.add_argument("--budget", type=int, default=200_000)
    ap.add_argument("--hooks", default=os.path.join(COMMON, "init_hooks.py")); ap.add_argument("--span", type=int, default=6, help="k from best_known to best_known+span")
    a = ap.parse_args(); graphs = a.graphs.split(",")
    jobs = [(g, k, a.budget, a.hooks) for g in graphs for k in range(BEST_KNOWN[g], BEST_KNOWN[g] + a.span + 1)]
    with ProcessPoolExecutor(min(32, len(jobs))) as ex: res = list(ex.map(one, jobs))
    print(f"budget = {a.budget:,} iterations; start hooks = {os.path.basename(a.hooks)}")
    print(f"{'graph':12}{'k':>4}{'best':>6}{'DSATUR':>8} | {'conflicts':>10}{'solved_at':>11}{'iters':>9}{'wall s':>8}")
    suggest = {}
    for g in graphs:
        rows = sorted([r for r in res if r[0] == g], key=lambda r: r[1])
        for _, k, bc, sa, it, t, k0 in rows:
            print(f"{g:12}{k:>4}{BEST_KNOWN[g]:>6}{k0:>8} | {bc:>10}{(sa if sa is not None else '-'):>11}{it:>9,}{t:>8.1f}")
            if g not in suggest and 1 <= bc <= 10: suggest[g] = k
    print("suggested targets (lowest k with 1..10 conflicts left):", suggest)

if __name__ == "__main__": main()
