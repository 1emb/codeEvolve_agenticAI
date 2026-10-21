#!/usr/bin/env python
# ===--------------------------------------------------------------------------------------===#
#
# Build the fixed-k "hook" graph coloring problem for CodeEvolve, paper EoH settings.
#
#   python make_problem_fixedk.py <name> --targets DSJC250.5=28,le450_25c=26,DSJC500.5=49 \
#       --budget 200000 --wall-cap 20 [--epochs 152 --islands 10 --force]
#
# input/ = evaluate.py (evaluate_fixedk), scoring.py, tabucol_framework.py, problem.json,
#          graphs/*.col, src/init_program.py (init_hooks)
#
# ===--------------------------------------------------------------------------------------===#

import argparse
import json
import os
import shutil
import sys

COMMON = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = os.path.dirname(COMMON)
sys.path.insert(0, COMMON)
from make_problem import BEST_KNOWN, ensure_graph, lm, write_yaml  # noqa: E402
from scoring import read_dimacs  # noqa: E402

FRAMEWORK_EXCERPT = '''    gamma = zeros((n, k))                 # gamma[v, c] = number of neighbours of v with colour c
    tabu_until = zeros((n, k))            # (v, c) is tabu while tabu_until[v, c] > it
    for it in 1 .. budget:
        own = gamma[rows, col]            # conflicts each vertex is in
        conflicting = vertices with own > 0
        v, c = select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng)
        conflicts += gamma[v, c] - gamma[v, col[v]]
        recolour v to c; update gamma for v's neighbours
        tabu_until[v, old_colour] = it + tenure(n, k, len(conflicting), it, rng)
        if conflicts < best_conf: remember this colouring (best_conf = conflicts)
        if conflicts == 0: stop, solved at iteration it'''


def sys_msg(instances, budget, wall_cap, eval_timeout, baseline):
    rows = "\n".join(
        f"  * {i['name']}: n = {i['n']}, m = {i['m']}, best known k = {i['best_known']}, target k = {i['target_k']}"
        for i in instances
    )
    base_rows = ", ".join(f"{b['name']} {b['conflicts']} conflicts" for b in baseline)
    return f"""# PROMPT-BLOCK-START
You are an expert in combinatorial optimization, metaheuristics, and graph coloring.
Your mission is to design better decision rules for TabuCol, the tabu search that looks for a legal k-colouring of a graph (no edge with both endpoints the same colour) by repeatedly recolouring one conflicting vertex.

PROBLEM CONTEXT:
- **Task**: Improve the two rules the fixed TabuCol framework calls on every iteration:
  * `select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng) -> (v, c)`: which conflicting vertex v to recolour, and to which colour c (c != col[v], 0 <= c < k).
  * `tenure(n, k, num_conflicting, it, rng) -> int`: for how many iterations the reverted move (v back to its old colour) stays tabu.
- **The framework (fixed, you cannot change it)** runs, for each training graph, at a fixed target number of colours k and a fixed budget of {budget:,} iterations:
```
{FRAMEWORK_EXCERPT}
```
- **Inputs to select_move**: `gamma` int32 (n, k) with gamma[v, c] = number of neighbours of v having colour c; `col` int64 (n,) current colours; `conflicting` int64 array of vertices with at least one same-coloured neighbour; `tabu_until` int64 (n, k); `it` current iteration (from 1); `conflicts` current conflicting edges; `best_conf` fewest conflicts seen so far; `k`; `rng` a `random.Random`.
- **Graph setup**: {len(instances)} DIMACS benchmark graphs (training set), one fixed-k search per graph:
{rows}
  The start colouring is DSATUR with the surplus colour classes recoloured at random (seeded, identical for every program).

EVALUATION:
- **Per-graph score** (lower is better): if the search reaches 0 conflicts, score = iterations_used / {budget:,} (a number in (0, 1], faster is better); otherwise score = number of conflicting edges left in the best colouring found (>= 1). A crash, an invalid move, or exceeding the {wall_cap:g} s wall-clock cap for one graph scores m + 1 (worse than any search).
- **Fitness** (maximize): -mean(score) over the training graphs. 0.0 means every target was solved instantly. Higher (closer to 0) is better.
- **best_conflicts**, **solved_at**, **iters**, **wall**, **status**, **reason**: per-graph details reported for every evaluation.
- Scores are deterministic: same rules, same score, on any machine. Only improvements to the rules move the score.

COMPUTATIONAL RESOURCES & IMPLEMENTATION GUIDELINES:
**Core packages**: numpy and the Python standard library only. No file, network, subprocess or threading access.
- Both rules are called once per iteration, {budget:,} times per graph; keep them vectorised (numpy over the `conflicting` rows) and avoid Python loops over all vertices or colours. The current select_move costs O(|conflicting| * k) per call.
- Use only `rng` for randomness (never `random`, `numpy.random` or the clock); the evaluation must be reproducible.
- You may keep state between calls (module-level variables, e.g. for reactive tenure or stagnation detection); the module is imported fresh for every evaluation and the three graphs are evaluated in sequence, so reset state when `it == 1`.
- Ideas with a track record on these graphs: random tie-breaking among equal best moves, reactive or adaptive tenure (longer when stagnating), occasionally sampling a non-best move, restricting candidates to a subset of conflicting vertices, aspiration variants, diversification after many non-improving iterations, and moves that take the colour of the conflicting neighbour into account.

COMPUTATIONAL BUDGET:
- **Time limit**: {wall_cap:g} s wall-clock per graph, {eval_timeout} s maximum total execution time.
- **Memory limit**: 2 GB.

KNOWN BASELINES:
- The starting rules (classic TabuCol: best non-tabu move, tenure 0.6 * |conflicting| + rand(10)) leave {base_rows} within the budget, fitness {baseline[0]['fitness']:.3f}.
- The target colour counts are at, or 1-2 above, the best known for each graph, which were found by much longer runs of tabu search and memetic algorithms; reaching 0 conflicts on a target is a real result.
# PROMPT-BLOCK-END
"""


def base_config(instances, budget, wall_cap, eval_timeout, epochs, islands, baseline):
    return {
        "SYS_MSG": sys_msg(instances, budget, wall_cap, eval_timeout, baseline),
        "CODEBASE_PATH": "src/",
        "INIT_FILE_DATA": {"filename": "init_program.py", "language": "python"},
        "EVAL_FILE_NAME": "evaluate.py",
        "BUDGET_CONFIG": {"eval_timeout": eval_timeout, "max_mem_bytes": 2_000_000_000, "resource_check_interval_s": 0.1},
        "EVOLVE_CONFIG": {
            "fitness_key": "fitness",
            "num_epochs": epochs, "ckpt": 10, "max_size": None, "init_pop": 10,
            "exploration_rate": 0.3,
            "selection": {"policy": "roulette", "kwargs": {"roulette_by_rank": True}},
            "early_stopping_rounds": None,
            "num_islands": islands,
            "migration": {"topology": "ring", "interval": 30, "rate": 0.1},
            "meta_prompting": True, "use_map_elites": True,
            "num_inspirations": 2,
            "max_chat_depth": 3,
            "markers": {"mp_start_marker": "# PROMPT-BLOCK-START", "mp_end_marker": "# PROMPT-BLOCK-END",
                        "evolve_start_marker": "# EVOLVE-BLOCK-START", "evolve_end_marker": "# EVOLVE-BLOCK-END"},
            "exploration_scheduler": {"type": "PlateauScheduler",
                                      "kwargs": {"min_value": 0.2, "max_value": 0.5, "plateau_threshold": 5,
                                                 "increase_factor": 1.05, "decrease_factor": 0.95}},
        },
        "MAP_ELITES": {
            "elite_map_type": "cvt",
            "elite_map_kwargs": {"num_centroids": 30, "num_init_samples": 1000, "max_iter": 300, "tolerance": 1e-4},
            "features": [
                {"name": "fitness", "min_val": -30.0, "max_val": 0.0},
                {"name": "eval_time", "min_val": 0, "max_val": eval_timeout},
            ],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--targets", required=True, help="comma-separated instance=target_k")
    ap.add_argument("--budget", type=int, default=200_000)
    ap.add_argument("--wall-cap", type=float, default=20.0)
    ap.add_argument("--epochs", type=int, default=152)
    ap.add_argument("--islands", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    targets = [(s.split("=")[0].strip(), int(s.split("=")[1])) for s in a.targets.split(",") if s.strip()]
    prob = os.path.join(PROBLEMS, a.name)
    inp = os.path.join(prob, "input")
    if os.path.exists(prob) and not a.force:
        sys.exit(f"{prob} exists; use --force to overwrite")
    os.makedirs(os.path.join(inp, "src"), exist_ok=True)
    os.makedirs(os.path.join(inp, "graphs"), exist_ok=True)

    instances = []
    for name, tk in targets:
        path = ensure_graph(name)
        n, edges = read_dimacs(path)
        shutil.copy(path, os.path.join(inp, "graphs", f"{name}.col"))
        instances.append({"name": name, "file": f"graphs/{name}.col", "n": n, "m": len(edges),
                          "best_known": BEST_KNOWN[name], "target_k": tk})

    eval_timeout = int(a.wall_cap * len(instances)) + 15
    shutil.copy(os.path.join(COMMON, "evaluate_fixedk.py"), os.path.join(inp, "evaluate.py"))
    shutil.copy(os.path.join(COMMON, "scoring.py"), inp)
    shutil.copy(os.path.join(COMMON, "tabucol_framework.py"), inp)
    shutil.copy(os.path.join(COMMON, "init_hooks.py"), os.path.join(inp, "src", "init_program.py"))
    with open(os.path.join(inp, "problem.json"), "w") as f:
        json.dump({"name": a.name, "budget": a.budget, "wall_cap": a.wall_cap, "instances": instances}, f, indent=2)

    # baseline: run the evaluator once on the starting hooks so the prompt can quote real numbers
    import subprocess
    out = os.path.join(inp, "baseline.json")
    subprocess.run([sys.executable, os.path.join(inp, "evaluate.py"), os.path.join(inp, "src", "init_program.py"), out],
                   check=True, cwd=inp)
    with open(out) as f:
        b = json.load(f)
    baseline = [{"name": i["name"], "conflicts": c, "fitness": b["fitness"]} for i, c in zip(instances, b["best_conflicts"])]
    os.remove(out)

    cfg = base_config(instances, a.budget, a.wall_cap, eval_timeout, a.epochs, a.islands, baseline)
    qwen = dict(cfg, EXPLORATION_ENSEMBLE=[lm(a.model, 0.7)], EXPLOITATION_ENSEMBLE=[lm(a.model, 0.7)],
                SAMPLER_AUX_LM=lm(a.model, 0.7))
    mock = dict(cfg, EXPLORATION_ENSEMBLE=[{"model_name": "MOCK"}], EXPLOITATION_ENSEMBLE=[{"model_name": "MOCK"}],
                SAMPLER_AUX_LM={"model_name": "MOCK"})
    write_yaml(os.path.join(prob, "config_qwen.yaml"), qwen)
    write_yaml(os.path.join(prob, "config_mock.yaml"), mock)
    print(f"built {prob}: targets={dict(targets)} budget={a.budget:,} wall_cap={a.wall_cap:g}s eval_timeout={eval_timeout}s "
          f"epochs={a.epochs} islands={a.islands}")
    print(f"baseline: fitness={b['fitness']:.4f} conflicts={b['best_conflicts']} wall={b['wall']}")


if __name__ == "__main__":
    main()
