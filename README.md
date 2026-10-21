# CodeEvolve for graph colouring: evolving TabuCol decision rules

This package contains the problem code, the best evolved program, the raw run
outputs and the instructions to reproduce the `gc_fixedk` experiment
(run finished 2026-10-21, 4 h 43 min, 10 islands, 152 epochs).

## 0. Read this first: what this experiment is and is not

* **The LLM did not invent a colouring algorithm.** It was given the classic
  TabuCol search (Hertz and de Werra, 1987) and could only rewrite its two
  decision rules, `select_move` and `tenure`. The search loop, start colouring,
  data structures and stopping rule are fixed code the LLM never saw as editable.
  The honest description of the result is "LLM-tuned TabuCol rules", not
  "LLM-discovered heuristic".
* **The training targets were calibrated to the seed.** The target k for each
  graph was chosen with `code/calibrate_fixedk.py` as the smallest k at which
  the seed rules fail by only 1 to 10 conflicts. This gives the search a usable
  gradient, but it also means a small tweak to the seed looks like a large gain
  on the training fitness. Training fitness is not comparable to any published
  number.
* **The LLM was given hints.** The system prompt in
  `problem/gc_fixedk/config_qwen.yaml` lists known ideas (reactive tenure,
  random tie-breaking, sampling non-best moves, diversification). The evolved
  program uses those ideas. This is guidance from the literature, not discovery.
* **The seed is a deliberately plain baseline.** It is the 1987 rule set without
  random tie-breaking or reactive tenure. A properly tuned TabuCol would likely
  match the evolved rules, so part of the measured gain comes from the seed
  being weak.
* **The gain does not transfer.** On the held-out test (section 5.2) the evolved
  rules save one colour on one graph, lose one colour on another, and tie
  elsewhere. In aggregate they are no better than the seed.
* **Nothing here approaches the state of the art.** Best-known DIMACS values
  (47 on DSJC500.5, 82 on DSJC1000.5) come from memetic algorithms such as HEAD
  running for hours in compiled code. This 60 s Python TabuCol lands at 52 and
  93, at the bottom of the published range, with or without the evolved rules.
* **The design was not taken from a published graph-colouring result.** It
  adapts the style of the Evolution of Heuristics paper (Liu et al., ICML 2024),
  which evaluated bin packing, TSP and flow shop, not graph colouring. We know of
  no published LLM-evolution result on DIMACS colouring to compare with.
* **The two earlier whole-program runs were weaker still.** DSJC125.5 run1
  started from DSATUR greedy only (the LLM added its own tabu search, reaching
  21 colours vs 17 best known). gc_train3 run1 started from DSATUR + TabuCol
  and reached 29/26/52 vs 28/25/47. The switch to the fixed-framework setup was
  made because of those results, which narrowed the research question.
* **The best evolved program has two defects** (section 6): it uses unseeded
  `numpy.random` instead of the supplied `rng`, and one of its branches is dead
  code. The scored behaviour is what is in the file, not what its comments say.

What is solid: the pipeline runs end to end, the training evaluation is
deterministic and independently verifies every reported colouring, every
colour count in the held-out table is a verified conflict-free colouring, and
every island beat the seed on the training objective.

## 1. What the experiment does

CodeEvolve (inter-co/science-codeevolve v0.3.1, arXiv 2510.14150) is an
LLM-driven evolutionary search over source code. Here the LLM is
Qwen/Qwen3-Coder-30B-A3B-Instruct served locally with vLLM.

Following the "Evolution of Heuristics" setup, the LLM does not write a whole
colouring algorithm. It only evolves the two decision rules that a fixed
TabuCol loop calls on every iteration:

* `select_move(gamma, col, conflicting, tabu_until, it, conflicts, best_conf, k, rng) -> (v, c)`
  which conflicting vertex to recolour and to which colour;
* `tenure(n, k, num_conflicting, it, rng) -> int`
  how long the reverted move stays tabu.

The loop itself (`code/tabucol_framework.py`) is fixed: DSATUR start colouring,
gamma matrix, tabu matrix, best-colouring tracking, 200,000 iterations per graph
at a fixed target number of colours k.

Training graphs and targets (chosen with `code/calibrate_fixedk.py` so that the
seed rules leave a few conflicts):

| graph      | n   | m     | best known k | target k |
|------------|-----|-------|--------------|----------|
| DSJC250.5  | 250 | 15668 | 28           | 28       |
| le450_25c  | 450 | 17343 | 25           | 26       |
| DSJC500.5  | 500 | 62624 | 47           | 49       |

## 2. Package layout

```
README.md                        this file
code/                            all problem code (nothing here is upstream CodeEvolve)
  tabucol_framework.py           fixed TabuCol loop + decreasing-k wrapper (not evolved)
  init_hooks.py                  seed program: classic TabuCol rules
  evaluate_fixedk.py             evaluator CodeEvolve calls: python evaluate.py <prog.py> <out.json>
  scoring.py                     DIMACS reader + shared helpers
  make_problem_fixedk.py         builds a problem directory + CodeEvolve config + system prompt
  calibrate_fixedk.py            runs seed rules at candidate k to pick target k per graph
  heldout_hooks.py               held-out test: colours used under a wall-clock limit
  run.sh                         launches CodeEvolve on a problem directory
  serve_qwen.sh                  starts the vLLM server
  make_problem.py, evaluate.py, init_program.py      DSJC125.5 run1: whole program, seed = DSATUR only
  make_problem_multi.py, evaluate_multi.py, init_tabucol.py   gc_train3 run1: whole program, seed = DSATUR + TabuCol
                                     (neither is used by gc_fixedk)
problem/gc_fixedk/               the assembled problem directory that was run
  input/evaluate.py, scoring.py, tabucol_framework.py, problem.json, graphs/*.col, src/init_program.py
  config_qwen.yaml               CodeEvolve config incl. system prompt
results/run1/
  best_overall.py                best evolved program (island 6)
  island_<i>/best_sol.py         best program of each island
  island_<i>/best_prompt.txt     the prompt that produced it
  run_metadata.json              best solution at every checkpoint epoch
  run_history.log                start/finish stamps
  eval/init.json, eval/isl_<i>.json   evaluator output for the seed and every island best
  heldout.txt                    held-out comparison seed vs evolved
data/dimacs/*.col                graphs used by the held-out test
```

## 3. How to run

Requirements: Python 3.12, numpy, PyYAML, CodeEvolve installed as the
`codeevolve` command (clone https://github.com/inter-co/science-codeevolve and
`pip install -e .`), and an OpenAI-compatible LLM endpoint.

The scripts assume this layout, adjust the paths at the top of the scripts
otherwise:

```
~/codeEvolve/codeEv/                       CodeEvolve checkout
~/codeEvolve/codeEv/problems/graph_coloring/common/   <- code/ from this package
~/codeEvolve/data/dimacs/                  <- data/dimacs from this package
~/codeEvolve/experiments/                  run outputs
```

### 3.1 Serve the model

```bash
bash code/serve_qwen.sh              # one H100, port 8000, bf16
export API_BASE=http://localhost:8000/v1 API_KEY=EMPTY
```

Any OpenAI-compatible server works. Change `model_name` in
`problem/gc_fixedk/config_qwen.yaml` if you use a different model.

### 3.2 Build the problem (already done, output is `problem/gc_fixedk`)

```bash
cd ~/codeEvolve/codeEv/problems/graph_coloring/common
python make_problem_fixedk.py gc_fixedk \
    --targets DSJC250.5=28,le450_25c=26,DSJC500.5=49 \
    --budget 200000 --wall-cap 20 --epochs 152 --islands 10
```

### 3.3 Evolve

```bash
cd ~/codeEvolve/codeEv
bash problems/graph_coloring/common/run.sh gc_fixedk qwen run2 2>&1 | tee ~/codeEvolve/gc_fixedk_run2.log
```

Output goes to `~/codeEvolve/experiments/graph_coloring/gc_fixedk/run2/`.
CodeEvolve writes `island_<i>/best_sol.py` and `run_metadata.json` but no
single overall best file. Pick the island with the highest `fitness` in the
final `run_metadata.json` entry and copy its `best_sol.py` to `best_overall.py`.
The run takes about 4.7 h on 128 CPU cores with one H100 serving the model.

### 3.4 Score any program on the training set (no LLM needed, about 30 s)

```bash
python problem/gc_fixedk/input/evaluate.py results/run1/best_overall.py out.json
python -c "import json; print(json.load(open('out.json')))"
```

### 3.5 Held-out test (no LLM needed)

```bash
cd code
python heldout_hooks.py init_hooks.py ../results/run1/best_overall.py \
    --graphs DSJC125.5,DSJC250.5,le450_25c,DSJC500.5,DSJC1000.5,queen8_8,myciel5 --time-limit 10
python heldout_hooks.py init_hooks.py ../results/run1/best_overall.py \
    --graphs DSJC125.5,DSJC250.5,le450_25c,DSJC500.5,DSJC1000.5 --time-limit 60
```

## 4. Fitness definition

For each training graph the rules run 200,000 iterations at the target k.

* solved (0 conflicts reached): `score = iterations_used / 200000`, in (0, 1]
* unsolved: `score = conflicting edges left in the best colouring`, >= 1
* crash, invalid move, or over the 20 s per-graph wall cap: `score = m + 1`

`fitness = -mean(score)` over the three graphs. CodeEvolve maximises it.
0 would mean every target solved instantly. The evaluation is deterministic
(seeded start colouring and rng, single-threaded numpy).

## 5. Results

### 5.1 Training objective

| program                      | fitness | DSJC250.5 k=28 | le450_25c k=26      | DSJC500.5 k=49 |
|------------------------------|---------|----------------|---------------------|----------------|
| seed (classic TabuCol)       | -5.333  | 2 conflicts    | 6 conflicts         | 8 conflicts    |
| best evolved (island 6)      | -1.795  | 3 conflicts    | solved, iter 76,816 | 2 conflicts    |
| island 7                     | -1.957  | 1 conflict     | solved, iter 174,483| 4 conflicts    |
| island 2                     | -2.322  | solved, iter 192,934 | 1 conflict    | 5 conflicts    |
| island 1                     | -2.333  | 3              | 1                   | 3              |
| island 9                     | -2.412  | 1              | solved, iter 46,886 | 6              |
| island 0                     | -2.567  | 1              | solved, iter 140,474| 6              |
| island 4                     | -2.835  | 1              | solved, iter 101,002| 7              |
| island 8                     | -2.938  | 2              | solved, iter 162,984| 6              |
| islands 3, 5                 | -3.667  | 2              | 1 / 3               | 8 / 6          |

Every island beat the seed. The global best improved at epochs 7, 11, 55, 72,
96, 117 and 126, then did not move for the last 26 epochs (see
`run_metadata.json`).

### 5.2 Held-out test (colours used by the decreasing-k wrapper, best known in parentheses)

```
time limit 10 s per graph
graph              seed    evolved
DSJC125.5  (17)      17         17
DSJC250.5  (28)      29         29
le450_25c  (25)      28         27
DSJC500.5  (47)      51         52
DSJC1000.5 (82)      96         95
queen8_8    (9)       9          9
myciel5     (6)       6          6

time limit 60 s per graph
DSJC125.5  (17)      17         17
DSJC250.5  (28)      29         29
le450_25c  (25)      27         26
DSJC500.5  (47)      51         52
DSJC1000.5 (82)      93         93
```

### 5.3 Against the state of the art (60 s per graph, verified conflict-free colourings)

| graph      | best known | seed TabuCol | evolved rules | published range for other methods |
|------------|------------|--------------|---------------|-----------------------------------|
| DSJC125.5  | 17         | 17           | 17            | solved by everything              |
| DSJC250.5  | 28         | 29           | 29            | 28 is routine for memetic methods |
| le450_25c  | 25         | 27           | 26            | 25 for most memetic methods       |
| DSJC500.5  | 47 (HEAD)  | 51           | 52            | 48 to 52                          |
| DSJC1000.5 | 82 (HEAD)  | 93           | 93            | 82 to 94                          |

Sources: Moalic and Gondran, "The new memetic algorithm HEAD for graph
coloring", EvoCOP 2015 / arXiv 1401.2184; Zheng et al., arXiv 2109.05948
(notes that 47 on DSJC500.5 is reached only occasionally and with tuned
parameters); comparison table in arXiv 2303.09632.

## 6. How to interpret the results

* **The evolution worked on its own objective.** Fitness went from -5.33 to
  -1.79. The best program solves le450_25c at k=26 (one above best known) in
  77k of 200k iterations and leaves only 2 conflicts on DSJC500.5 at k=49.
  The mechanisms it found are the ones the literature uses: adaptive tenure
  that grows under stagnation, and switching to softmax or epsilon-greedy move
  sampling after a long non-improving stretch.
* **Different islands solved different graphs.** Island 2 cracked DSJC250.5 at
  k=28 (the best known value), islands 0, 4, 6, 7, 8, 9 cracked le450_25c at
  k=26, none cracked DSJC500.5. No single rule set solves all three, so the
  mean-score fitness is rewarding specialisation.
* **Generalisation is weak.** On held-out graphs the evolved rules save one
  colour on le450_25c, tie on the small graphs and on DSJC1000.5, and lose one
  colour on DSJC500.5. The rules are tuned to one iteration budget and one
  seeded start colouring, which is not what a time-limited decreasing-k search
  needs. Treat the training fitness as evidence of search power, not as a
  better colouring heuristic.
* **Compared with the earlier whole-program experiments** (DSJC125.5 alone:
  21 colours vs 17 best known; three graphs with gap-percent fitness: 29/26/52
  vs 28/25/47), the fixed-framework approach is clearly the better use of the
  LLM.

Two defects in the best evolved program to keep in mind:

1. It uses `np.random.choice` instead of the supplied `rng`. The evaluator seeds
   numpy so training scores are reproducible, but `heldout_hooks.py` does not,
   so held-out colour counts can differ by one between runs.
2. The exploration flag compares `conflicts` with `_last_conflicts` after
   overwriting the latter, so that branch never fires. The behaviour that was
   scored is what is in the file, not what the comments claim.

## 7. Suggested next step

The project owner's stated goal is to evolve a whole colouring program, not
TabuCol rules. The gc_fixedk setup does not meet that goal and should be read
as a pipeline test. A whole-program run needs a decision on the seed
(DSATUR-only is closest to "from scratch"; DSATUR + TabuCol is more likely to
reach best-known values), a budget-independent objective, and several hundred
epochs.

For the fixed-framework setup itself:

Evaluate each candidate on several seeded start colourings or two iteration
budgets, so the rules cannot overfit one search trajectory, and add a
reproducibility check (fail the evaluation if the program imports
`numpy.random` or `random`).
