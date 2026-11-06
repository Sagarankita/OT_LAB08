from itertools import combinations

import pytest

from optimization_lab.test_selection import (
    count_meaningful_python_lines,
    select_greedy,
    select_ilp,
    select_naive,
)


def problem():
    coverage = {
        "a": {("x.py", 1), ("x.py", 2)},
        "b": {("x.py", 2), ("x.py", 3)},
        "c": {("x.py", 1)},
        "d": {("x.py", 3)},
    }
    durations = {"a": 4.0, "b": 4.0, "c": 1.0, "d": 1.0}
    target = {("x.py", 1), ("x.py", 2), ("x.py", 3)}
    return coverage, durations, target


def test_all_selectors_preserve_target_and_ilp_is_exact():
    coverage, durations, target = problem()
    results = [
        select_naive(coverage, durations, target),
        select_greedy(coverage, durations, target),
        select_ilp(coverage, durations, target),
    ]
    assert all(result.covered_lines == len(target) for result in results)

    brute_force = min(
        sum(durations[test] for test in subset)
        for size in range(1, len(coverage) + 1)
        for subset in combinations(coverage, size)
        if target <= set().union(*(coverage[test] for test in subset))
    )
    assert results[-1].objective_seconds == pytest.approx(brute_force)


def test_problem_validation_rejects_uncoverable_target():
    coverage, durations, target = problem()
    target.add(("missing.py", 1))
    with pytest.raises(ValueError, match="uncoverable"):
        select_greedy(coverage, durations, target)


def test_meaningful_line_counter_excludes_blank_and_comments():
    assert count_meaningful_python_lines("# comment\n\nx = 1\n  # other\nreturn x\n") == 2
