"""Tests for the graph coloring scoring function. Run with pytest or directly."""
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import failure_fitness, fitness_formula, read_dimacs, score  # noqa: E402

# Triangle plus a pendant vertex: 1-2, 1-3, 2-3, 3-4. n=4, m=4, chi=3.
N = 4
EDGES = [(1, 2), (1, 3), (2, 3), (3, 4)]
T = 10.0


def test_valid_fitness_formula():
    r = score(N, EDGES, [1, 2, 3, 1], elapsed=0.0, time_limit=T)
    assert r["valid"] and r["k"] == 3 and r["conflicts"] == 0
    assert r["fitness"] == (N + 1 - 3) == 2.0
    r = score(N, EDGES, [1, 2, 3, 1], elapsed=2.5, time_limit=T)
    assert abs(r["fitness"] - (2.0 - 0.25)) < 1e-12


def test_fewer_colors_beats_any_time():
    slow_3 = score(N, EDGES, [1, 2, 3, 1], elapsed=9.99, time_limit=T)
    fast_4 = score(N, EDGES, [1, 2, 3, 4], elapsed=0.0, time_limit=T)
    assert slow_3["fitness"] > fast_4["fitness"]


def test_same_k_faster_wins():
    a = score(N, EDGES, [1, 2, 3, 1], elapsed=1.0, time_limit=T)
    b = score(N, EDGES, [1, 2, 3, 2], elapsed=5.0, time_limit=T)
    assert a["k"] == b["k"] == 3
    assert a["fitness"] > b["fitness"]


def test_color_labels_are_arbitrary():
    r = score(N, EDGES, [10, -5, 0, 10], elapsed=0.0, time_limit=T)
    assert r["valid"] and r["k"] == 3


def test_improper_is_negative_and_ordered():
    one = score(N, EDGES, [1, 1, 2, 1], elapsed=0.0, time_limit=T)    # conflict on 1-2
    three = score(N, EDGES, [1, 1, 1, 2], elapsed=0.0, time_limit=T)  # 1-2, 1-3, 2-3
    assert one["status"] == "improper" and one["conflicts"] == 1
    assert three["conflicts"] == 3
    assert one["fitness"] == (1 - 1) * (N + 1) - 2 - 0.0 == -2.0     # k = 2
    assert three["fitness"] == (1 - 3) * (N + 1) - 2 - 0.0 == -12.0   # k = 2
    assert one["fitness"] > three["fitness"]


def test_improper_conflict_then_k_then_time():
    k2 = score(N, EDGES, [1, 1, 2, 1], elapsed=0.0, time_limit=T)   # 1 conflict, k=2
    k3 = score(N, EDGES, [1, 1, 2, 3], elapsed=0.0, time_limit=T)   # 1 conflict, k=3
    assert k2["conflicts"] == k3["conflicts"] == 1
    assert k2["fitness"] > k3["fitness"]
    fast = score(N, EDGES, [1, 1, 2, 3], elapsed=0.0, time_limit=T)
    slow = score(N, EDGES, [1, 1, 2, 3], elapsed=9.0, time_limit=T)
    assert fast["fitness"] > slow["fitness"]
    # k and time can never outweigh one conflict
    best_3conf = score(N, EDGES, [1, 1, 1, 2], elapsed=0.0, time_limit=T)
    worst_1conf = score(N, EDGES, [1, 1, 2, 3], elapsed=9.999, time_limit=T)
    assert worst_1conf["fitness"] > best_3conf["fitness"]


def test_failures_are_worst():
    m = len(EDGES)
    expected = failure_fitness(N, m)
    assert expected == -(m + 1) * (N + 1) - 1 == -26.0
    cases = {
        "wrong_len": score(N, EDGES, [1, 2, 3], 0.0, T),
        "float": score(N, EDGES, [1.0, 2, 3, 1], 0.0, T),
        "bool": score(N, EDGES, [True, 2, 3, 1], 0.0, T),
        "none": score(N, EDGES, None, 0.0, T),
        "error": score(N, EDGES, [1, 2, 3, 1], 0.0, T, error="ZeroDivisionError"),
        "timeout": score(N, EDGES, [1, 2, 3, 1], 10.0, T),
    }
    for name, r in cases.items():
        assert r["fitness"] == expected, name
        assert r["valid"] is False and r["k"] is None, name
    assert cases["timeout"]["status"] == "timeout"
    assert cases["error"]["status"] == "error"
    assert cases["wrong_len"]["status"] == "bad_output"
    all_bad = score(N, EDGES, [1, 1, 1, 1], 9.999, T)   # every edge conflicting
    assert all_bad["conflicts"] == m
    assert all_bad["fitness"] > expected


def test_valid_above_zero_above_improper_random():
    rng = random.Random(0)
    n = 30
    edges = sorted({(min(u, v), max(u, v)) for _ in range(120)
                    for u, v in [(rng.randint(1, n), rng.randint(1, n))] if u != v})
    m = len(edges)
    for _ in range(300):
        colors = [rng.randint(1, rng.randint(1, n)) for _ in range(n)]
        t = rng.uniform(0, T - 1e-9)
        r = score(n, edges, colors, t, T)
        if r["valid"]:
            assert 0 < r["fitness"] <= n
        else:
            assert -m * (n + 1) < r["fitness"] < -1
        assert r["fitness"] > failure_fitness(n, m)


def test_single_formula_covers_all_cases():
    for colors, elapsed in [([1, 2, 3, 1], 0.5), ([1, 1, 2, 3], 4.0), ([1, 1, 1, 1], 9.0)]:
        r = score(N, EDGES, colors, elapsed, T)
        assert r["fitness"] == fitness_formula(N, r["conflicts"], r["k"], elapsed / T)


def test_numpy_ints_accepted():
    try:
        import numpy as np
    except ImportError:
        return
    r = score(N, EDGES, np.array([1, 2, 3, 1]), 0.0, T)
    assert r["valid"] and r["k"] == 3


def test_read_dimacs():
    with tempfile.NamedTemporaryFile("w", suffix=".col", delete=False) as f:
        f.write("c comment\np edge 4 5\ne 1 2\ne 2 1\ne 2 3\ne 1 3\ne 3 4\ne 4 4\n")
        path = f.name
    n, edges = read_dimacs(path)
    os.unlink(path)
    assert n == 4
    assert edges == [(1, 2), (1, 3), (2, 3), (3, 4)]  # duplicate and self-loop dropped


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"{len(fns)} tests passed")
