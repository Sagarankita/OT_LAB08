"""Coverage-preserving test-suite selection algorithms for Track J."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


Line = tuple[str, int]


@dataclass(frozen=True)
class SelectionResult:
    method: str
    selected: tuple[str, ...]
    objective_seconds: float
    overhead_seconds: float
    covered_lines: int
    solver_status: str


def validate_problem(
    coverage: Mapping[str, set[Line]],
    durations: Mapping[str, float],
    target: set[Line],
) -> None:
    if not coverage:
        raise ValueError("coverage mapping must not be empty")
    if set(coverage) != set(durations):
        raise ValueError("coverage and duration test identifiers must match")
    if any(value < 0 for value in durations.values()):
        raise ValueError("durations must be non-negative")
    available = set().union(*coverage.values())
    missing = target - available
    if missing:
        raise ValueError(f"target contains {len(missing)} uncoverable lines")


def _result(
    method: str,
    selected: Sequence[str],
    coverage: Mapping[str, set[Line]],
    durations: Mapping[str, float],
    target: set[Line],
    started: float,
    status: str,
) -> SelectionResult:
    ordered = tuple(sorted(selected))
    covered = set().union(*(coverage[test] for test in ordered)) if ordered else set()
    if not target <= covered:
        raise AssertionError(f"{method} failed to preserve the coverage target")
    return SelectionResult(
        method=method,
        selected=ordered,
        objective_seconds=sum(durations[test] for test in ordered),
        overhead_seconds=perf_counter() - started,
        covered_lines=len(covered & target),
        solver_status=status,
    )


def select_naive(
    coverage: Mapping[str, set[Line]],
    durations: Mapping[str, float],
    target: set[Line],
) -> SelectionResult:
    """Deterministic lexical first-cover baseline."""
    validate_problem(coverage, durations, target)
    started = perf_counter()
    uncovered = set(target)
    selected: list[str] = []
    for test in sorted(coverage):
        if coverage[test] & uncovered:
            selected.append(test)
            uncovered -= coverage[test]
        if not uncovered:
            break
    return _result("naive", selected, coverage, durations, target, started, "deterministic")


def select_greedy(
    coverage: Mapping[str, set[Line]],
    durations: Mapping[str, float],
    target: set[Line],
) -> SelectionResult:
    """Weighted greedy set cover using newly covered lines per second."""
    validate_problem(coverage, durations, target)
    started = perf_counter()
    uncovered = set(target)
    remaining = set(coverage)
    selected: list[str] = []
    while uncovered:
        candidates = []
        for test in remaining:
            gain = len(coverage[test] & uncovered)
            if gain:
                cost = max(durations[test], 1e-9)
                candidates.append((gain / cost, gain, -cost, test))
        if not candidates:
            raise AssertionError("greedy selection became infeasible")
        test = max(candidates)[3]
        selected.append(test)
        uncovered -= coverage[test]
        remaining.remove(test)
    return _result("greedy", selected, coverage, durations, target, started, "feasible")


def select_ilp(
    coverage: Mapping[str, set[Line]],
    durations: Mapping[str, float],
    target: set[Line],
    *,
    time_limit_seconds: float = 60.0,
) -> SelectionResult:
    """Solve weighted set cover exactly as a binary integer linear program."""
    validate_problem(coverage, durations, target)
    started = perf_counter()
    tests = sorted(coverage)
    lines = sorted(target)
    line_index = {line: row for row, line in enumerate(lines)}
    rows: list[int] = []
    cols: list[int] = []
    for col, test in enumerate(tests):
        for line in coverage[test] & target:
            rows.append(line_index[line])
            cols.append(col)
    matrix = csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(lines), len(tests)))
    costs = np.array([max(durations[test], 1e-9) for test in tests])
    result = milp(
        c=costs,
        integrality=np.ones(len(tests)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, np.ones(len(lines)), np.full(len(lines), np.inf)),
        options={"time_limit": time_limit_seconds, "mip_rel_gap": 0.0},
    )
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(f"ILP failed: status={result.status}, message={result.message}")
    selected = [test for test, value in zip(tests, result.x) if value >= 0.5]
    status = "optimal" if result.status == 0 else "time_limit_feasible"
    return _result("ilp", selected, coverage, durations, target, started, status)


def count_meaningful_python_lines(text: str) -> int:
    """Count physical Python lines excluding blank and comment-only lines."""
    return sum(
        1
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
