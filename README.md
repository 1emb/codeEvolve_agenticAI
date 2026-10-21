# CodeEvolve for graph colouring: evolving TabuCol decision rules

This package contains the problem code, the best evolved program, the raw run
outputs and the instructions to reproduce the `gc_fixedk` experiment
(run finished 2026-10-21, 4 h 43 min, 10 islands, 152 epochs).

## 0. Read this first: what this experiment is and is not

* **The LLM did not invent a colouring algorithm.** It was given the classic
  TabuCol search (Hertz and de Werra, 1987) and could only rewrite its two
  decision rules, `select_move` and `tenure`. The search loop, start colouring,
  data structures and stopping rule are fixed code outside the editable block.
  The accurate description of the result is "LLM-tuned TabuCol rules".
* **The headline training score is the luckiest of eight random seeds.** The
  best program scored -1.79 on the one seed the evolution used. Re-scored under
  eight seeds it averages -3.82 (sd 1.08) against -5.63 (sd 0.68) for the
  starting rules. The gain over the starting rules is real but about half the
  size the single-seed number suggests. Details in section 5.4.
* **How the target colour counts were chosen.** TabuCol needs a fixed k per
  graph. The values used (28, 26, 49) are the smallest k at which the starting
  rules fail by 1 to 10 conflicts in 200,000 iterations, measured with
  `code/calibrate_fixedk.py` (section 5.5). The purpose was a starting point
  that is neither already solved nor hopeless. The consequence is that the
  targets depend on the starting rules, so a small change to those rules can
  look like a large gain. The starting-rule-independent alternative is
  k = best known for every graph, at which the starting rules leave 2, 11 and
  27 conflicts.
* **Run1 used a system prompt with hints written by the author.** The prompt
  listed ideas with a track record on these graphs (reactive tenure, random
  tie-breaking, sampling non-best moves, diversification), and the evolved
  program uses those ideas. The LLM cannot ask for anything; it receives the
  prompt on every call. The hints have since been removed from the prompt
  builder and from `problem/gc_fixedk/config_qwen.yaml`, so a new run starts
  without them. The prompt run1 actually used is preserved unchanged in
  `results/run1/config_qwen.yaml`. No run without hints has been made yet.
* **The starting rules are the simplest textbook TabuCol:** best non-tabu move,
  tenure 0.6 x (number of conflicting vertices) + random(0..9). Stronger
  published variants add random tie-breaking and reactive tenure. Evolution
  re-added such features, so part of the measured gain is recovering known
  improvements rather than finding new ones.
* **The gain barely transfers to colour counts.** On the held-out test
  (section 5.2) the only difference consistent across runs is one colour on
  le450_25c. Every other entry is within the one-colour run-to-run noise.
* **Nothing here approaches the state of the art.** Best-known DIMACS values
  (47 on DSJC500.5, 82 on DSJC1000.5) come from memetic algorithms such as HEAD
  running for hours in compiled code. This 60 s Python TabuCol lands at 51 to 52
  and 92 to 93, at the bottom of the published range, with or without the
  evolved rules (section 5.3).
* **The design was not taken from a published graph-colouring result.** It
  adapts the style of the Evolution of Heuristics paper (Liu et al., ICML 2024),
  which evaluated bin packing, TSP and flow shop, not graph colouring. We know of
  no published LLM-evolution result on DIMACS colouring to compare with.
* **The best evolved program had three defects; corrected copies are provided.**
  It sampled from unseeded `numpy.random`, one branch was dead code, and
  `tenure()` overwrote a variable `select_move` depends on.
  `results/run1/best_overall_rngfix.py` fixes only the randomness and
  `results/run1/best_overall_fixed.py` fixes all three. Neither scores better
  (section 5.4). The original is kept unchanged as the record of what evolution
  produced.

What is solid: the pipeline runs end to end, the training evaluation is
deterministic and independently verifies every reported colouring, every
colour count in the held-out table is a verified conflict-free colouring, and
every island beat the starting rules on the evolution seed.

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
  make_problem_fixedk.py         builds a problem directory + CodeEvolve config + system prompt (hint-free)
  calibrate_fixedk.py            runs seed rules at candidate k to pick target k per graph
  heldout_hooks.py               held-out test: colours used under a wall-clock limit
  run.sh                         launches CodeEvolve on a problem directory
  serve_qwen.sh                  starts the vLLM server
  make_problem.py                shared helpers: best-known table, graph download, yaml writer
problem/gc_fixedk/               the assembled problem directory that was run
  input/evaluate.py, scoring.py, tabucol_framework.py, problem.json, graphs/*.col, src/init_program.py
  config_qwen.yaml               CodeEvolve config incl. system prompt, regenerated without hints
results/run1/
  config_qwen.yaml               the config and system prompt run1 actually used (with the hints)
  best_overall.py                best evolved program (island 6), unchanged
  best_overall_rngfix.py         hand-corrected copy: randomness drawn from rng only
  best_overall_fixed.py          hand-corrected copy: all three defects fixed
  island_<i>/best_sol.py         best program of each island
  island_<i>/best_prompt.txt     the prompt that produced it
  run_metadata.json              best solution at every checkpoint epoch
  run_history.log                start/finish stamps
  eval/init.json, eval/isl_<i>.json   evaluator output for the seed and every island best
  eval/best_overall_*.json       evaluator output for the corrected copies
  eval/seeds/<prog>_s<seed>.json evaluator output under 8 evaluation seeds (section 5.4)
  heldout.txt                    held-out comparison, two runs
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

### 3.2 Build the problem (already done, output is `problem/gc_fixedk`; the builder now writes a prompt without hints)

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

Second run of the test, with the corrected copies included. The first run is
in `results/run1/heldout.txt`; it differs from this one by one colour on three
entries, which is the run-to-run noise of a wall-clock-limited search. A
difference of one colour below is not meaningful. The one difference that
holds in both runs and every variant is le450_25c.

```
time limit 10 s per graph
graph              seed   evolved   rngfix   fixed
DSJC125.5  (17)      17        18       17      17
DSJC250.5  (28)      29        29       29      29
le450_25c  (25)      28        27       27      26
DSJC500.5  (47)      51        52       51      52
DSJC1000.5 (82)      96        95       95      94
queen8_8    (9)       9         9        9       9
myciel5     (6)       6         6        6       6

time limit 60 s per graph
DSJC125.5  (17)      17        17       17      17
DSJC250.5  (28)      29        29       29      29
le450_25c  (25)      27        26       26      26
DSJC500.5  (47)      51        52       51      51
DSJC1000.5 (82)      93        92       92      92
```

### 5.3 Against the state of the art (60 s per graph, verified conflict-free colourings, two runs)

| graph      | best known | seed TabuCol | evolved rules (all variants) | published range for other methods |
|------------|------------|--------------|------------------------------|-----------------------------------|
| DSJC125.5  | 17         | 17           | 17                           | solved by everything              |
| DSJC250.5  | 28         | 29           | 29                           | 28 is routine for memetic methods |
| le450_25c  | 25         | 27           | 26                           | 25 for most memetic methods       |
| DSJC500.5  | 47 (HEAD)  | 51           | 51 to 52                     | 48 to 52                          |
| DSJC1000.5 | 82 (HEAD)  | 93           | 92 to 93                     | 82 to 94                          |

Sources: Moalic and Gondran, "The new memetic algorithm HEAD for graph
coloring", EvoCOP 2015 / arXiv 1401.2184; Zheng et al., arXiv 2109.05948
(notes that 47 on DSJC500.5 is reached only occasionally and with tuned
parameters); comparison table in arXiv 2303.09632.

### 5.4 Corrected copies and the effect of the evaluation seed

The evolved program had three defects (listed in section 6). Two corrected
copies are provided; the original is kept unchanged as the record of what
evolution produced.

* `best_overall_rngfix.py`: only the randomness is fixed (softmax sampling
  draws from the supplied `rng`). Same move distribution as the original,
  different random stream.
* `best_overall_fixed.py`: all three defects fixed to the evident intent of
  the code.

Training fitness under eight evaluation seeds. Seed 0 is the one the evolution
used; the evaluator was run with `evaluate.SEED` set to 0..7, which changes the
start colouring and the random stream.

| program                 | seed 0 | mean of 8 | sd   | range          | le450_25c solved at k=26 (of 8) |
|-------------------------|--------|-----------|------|----------------|---------------------------------|
| starting rules          | -5.33  | -5.63     | 0.68 | -6.67 to -4.67 | 0                               |
| best evolved (original) | -1.79  | -3.82     | 1.08 | -5.00 to -1.79 | 5                               |
| rngfix                  | -5.00  | -4.36     | 0.55 | -5.17 to -3.67 | 5 (and DSJC250.5 once)          |
| fixed                   | -4.00  | -4.56     | 0.68 | -5.33 to -3.92 | 3                               |

Reading:

* The headline -1.79 is the best of the eight seeds for the original program
  and the only seed the evolution ever scored. The same rules with a different
  random stream (rngfix on seed 0) score -5.00.
* Averaged over seeds the evolved rules beat the starting rules by about 1.8
  fitness points, almost all of it from solving le450_25c at k = 26 in some
  seeds. The worst evolved seed (-5.00) is about equal to the best
  starting-rules seed (-4.67). The improvement is real but much smaller than
  the single-seed number.
* Fixing the logic defects does not improve the rules. The behaviour evolution
  selected is the one with the defects in it.
* Per-seed outputs are in `results/run1/eval/seeds/`.

### 5.5 Starting rules at k = best known, +1, +2 (the calibration data)

`python code/calibrate_fixedk.py --graphs DSJC250.5,le450_25c,DSJC500.5 --budget 200000 --span 2`

| graph      | k = best known   | k + 1                 | k + 2                 |
|------------|------------------|-----------------------|-----------------------|
| DSJC250.5  | 28: 2 conflicts  | 29: solved at 177,231 | 30: solved at 16,519  |
| le450_25c  | 25: 11 conflicts | 26: 6 conflicts       | 27: solved at 47,802  |
| DSJC500.5  | 47: 27 conflicts | 48: 17 conflicts      | 49: 8 conflicts       |

The targets used (28, 26, 49) are the lowest k per graph with 1 to 10
conflicts left. Using k = best known everywhere would make the problem
independent of the starting rules and the fitness would still be graded
(2, 11, 27 conflicts), but reaching 0 conflicts on DSJC500.5 at k = 47 in
200,000 iterations of plain TabuCol is unrealistic, so the signal would come
from conflict counts only.

## 6. How to interpret the results

* **The evolution improved its own objective, by less than the headline
  number.** On the seed it was scored with, fitness went from -5.33 to -1.79.
  Averaged over eight seeds it went from -5.63 to -3.82. The mechanisms it
  found are the ones the literature uses: adaptive tenure that grows under
  stagnation, and switching to softmax or epsilon-greedy move sampling after a
  long non-improving stretch.
* **Single-seed evaluation let the search select on luck.** The best program's
  score on the evolution seed is its best of eight seeds. Any future run should
  average several seeds per evaluation.
* **Different islands solved different graphs.** Island 2 cracked DSJC250.5 at
  k=28 (the best known value), islands 0, 4, 6, 7, 8, 9 cracked le450_25c at
  k=26, none cracked DSJC500.5. No single rule set solves all three, so the
  mean-score fitness is rewarding specialisation.
* **Generalisation is weak.** On held-out graphs the only consistent difference
  is one colour on le450_25c. Everything else is within the one-colour
  run-to-run noise. Treat the training fitness as evidence of search power on
  one fixed trajectory, not as a better colouring heuristic.

Defects found in the best evolved program (corrected copies in section 5.4):

1. It sampled with `numpy.random.choice` instead of the supplied `rng`. The
   evaluator seeds numpy, so the training score was reproducible, but only for
   that one stream.
2. The exploitation branch compared `conflicts` with `_last_conflicts` after
   overwriting the latter, so it never fired.
3. `tenure()` overwrote `_last_conflicts`, an edge count that `select_move`
   uses for stagnation detection, with a vertex count, so the stagnation
   counter in `select_move` was almost always reset. Found while fixing the
   first two; not listed in the earlier version of this README.

The scored behaviour is what is in the original file, not what its comments
describe.


## 7. Suggested next step

The next step is a whole-program run: the program receives a graph and a time
limit and returns a colouring, with no target k prescribed. The gc_fixedk setup
should be read as a pipeline test.

For the fixed-framework setup itself: evaluate each candidate on several seeded
start colourings or two iteration budgets, so the rules cannot overfit one
search trajectory; add a reproducibility check (fail the evaluation if the
program imports `numpy.random` or `random`); and run with the hint-free prompt
now produced by the builder.
