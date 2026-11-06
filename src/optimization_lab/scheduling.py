"""Deterministic resource-constrained job-scheduling experiments for Track E."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Job:
    duration: int
    release: int
    predecessors: tuple[int, ...] = ()


@dataclass(frozen=True)
class ScheduledJob:
    job: int
    worker: int
    start: int
    finish: int


@dataclass(frozen=True)
class ScheduleSolution:
    entries: tuple[ScheduledJob, ...]
    makespan: int
    details: dict[str, object]


def validate_instance(jobs: Sequence[Job], workers: int) -> None:
    if not jobs:
        raise ValueError("jobs must be non-empty")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    for j, job in enumerate(jobs):
        if isinstance(job.duration, bool) or not isinstance(job.duration, int) or job.duration <= 0:
            raise ValueError(f"job {j} duration must be a positive integer")
        if isinstance(job.release, bool) or not isinstance(job.release, int) or job.release < 0:
            raise ValueError(f"job {j} release must be a non-negative integer")
        if tuple(sorted(set(job.predecessors))) != job.predecessors:
            raise ValueError(f"job {j} predecessors must be sorted and unique")
        if any(p < 0 or p >= j for p in job.predecessors):
            raise ValueError("predecessors must have smaller indices, defining a DAG")


def generate_instance(seed: int, count: int, edge_probability: float = 0.18,
                      max_predecessors: int = 3, duration_range: tuple[int, int] = (1, 9),
                      release_range: tuple[int, int] = (0, 6)) -> list[Job]:
    if count <= 0 or not 0 <= edge_probability <= 1 or max_predecessors < 0:
        raise ValueError("invalid generation settings")
    rng = random.Random(seed)
    jobs: list[Job] = []
    for j in range(count):
        candidates = [i for i in range(j) if rng.random() < edge_probability]
        if len(candidates) > max_predecessors:
            candidates = rng.sample(candidates, max_predecessors)
        # Give most non-root jobs a dependency, while retaining some parallel work.
        if j >= 2 and not candidates and rng.random() < 0.65:
            candidates = [rng.randrange(j)]
        jobs.append(Job(
            duration=rng.randint(*duration_range),
            release=rng.randint(*release_range),
            predecessors=tuple(sorted(candidates)),
        ))
    validate_instance(jobs, 1)
    return jobs


def validate_schedule(jobs: Sequence[Job], workers: int,
                      solution: ScheduleSolution) -> list[str]:
    """Return all constraint violations instead of failing after the first."""
    violations: list[str] = []
    if len(solution.entries) != len(jobs):
        violations.append("schedule must contain exactly one entry per job")
    by_job: dict[int, ScheduledJob] = {}
    for entry in solution.entries:
        if entry.job in by_job:
            violations.append(f"job {entry.job} appears more than once")
        by_job[entry.job] = entry
        if not 0 <= entry.job < len(jobs):
            violations.append(f"unknown job {entry.job}")
            continue
        if not 0 <= entry.worker < workers:
            violations.append(f"job {entry.job} has invalid worker")
        job = jobs[entry.job]
        if entry.start < job.release:
            violations.append(f"job {entry.job} starts before release")
        if entry.finish != entry.start + job.duration:
            violations.append(f"job {entry.job} has inconsistent finish")
    for j, job in enumerate(jobs):
        if j not in by_job:
            violations.append(f"job {j} is missing")
            continue
        for predecessor in job.predecessors:
            if predecessor in by_job and by_job[j].start < by_job[predecessor].finish:
                violations.append(f"precedence {predecessor}->{j} is violated")
    for worker in range(workers):
        assigned = sorted((e for e in solution.entries if e.worker == worker),
                          key=lambda e: (e.start, e.job))
        for left, right in zip(assigned, assigned[1:]):
            if right.start < left.finish:
                violations.append(f"worker {worker} overlaps jobs {left.job} and {right.job}")
    actual = max((entry.finish for entry in solution.entries), default=0)
    if solution.makespan != actual:
        violations.append("reported makespan does not match entries")
    return violations


def downstream_critical_lengths(jobs: Sequence[Job]) -> tuple[int, ...]:
    successors: list[list[int]] = [[] for _ in jobs]
    for j, job in enumerate(jobs):
        for predecessor in job.predecessors:
            successors[predecessor].append(j)
    lengths = [0] * len(jobs)
    for j in range(len(jobs) - 1, -1, -1):
        lengths[j] = jobs[j].duration + max((lengths[s] for s in successors[j]), default=0)
    return tuple(lengths)


def schedule_lower_bound(jobs: Sequence[Job], workers: int) -> int:
    validate_instance(jobs, workers)
    critical = downstream_critical_lengths(jobs)
    work_bound = math.ceil(sum(job.duration for job in jobs) / workers)
    release_chain_bound = max(job.release + critical[j] for j, job in enumerate(jobs))
    return max(work_bound, release_chain_bound)


def list_schedule(jobs: Sequence[Job], workers: int, priority: str) -> ScheduleSolution:
    validate_instance(jobs, workers)
    if priority not in {"fixed_order", "critical_path"}:
        raise ValueError("priority must be fixed_order or critical_path")
    critical = downstream_critical_lengths(jobs)
    worker_available = [0] * workers
    unscheduled = set(range(len(jobs)))
    by_job: dict[int, ScheduledJob] = {}
    entries: list[ScheduledJob] = []
    while unscheduled:
        ready = [j for j in unscheduled if all(p in by_job for p in jobs[j].predecessors)]
        if not ready:
            raise ValueError("instance contains a cycle")
        if priority == "fixed_order":
            job_id = min(ready)
        else:
            job_id = min(ready, key=lambda j: (-critical[j], -jobs[j].duration, j))
        job = jobs[job_id]
        predecessor_finish = max((by_job[p].finish for p in job.predecessors), default=0)
        candidates = []
        for worker in range(workers):
            start = max(job.release, predecessor_finish, worker_available[worker])
            candidates.append((start + job.duration, start, worker))
        finish, start, worker = min(candidates)
        entry = ScheduledJob(job_id, worker, start, finish)
        entries.append(entry)
        by_job[job_id] = entry
        worker_available[worker] = finish
        unscheduled.remove(job_id)
    solution = ScheduleSolution(tuple(sorted(entries, key=lambda e: e.job)),
                                max(worker_available), {"status": "heuristic"})
    assert not validate_schedule(jobs, workers, solution)
    return solution


def exhaustive_schedule(jobs: Sequence[Job], workers: int) -> ScheduleSolution:
    """Enumerate topological dispatch choices and worker assignments for tiny cases."""
    validate_instance(jobs, workers)
    if len(jobs) > 9:
        raise ValueError("exhaustive oracle is limited to at most nine jobs")
    incumbent = list_schedule(jobs, workers, "critical_path")
    best_makespan = incumbent.makespan
    best_entries = list(incumbent.entries)
    nodes = 0

    def visit(unscheduled: frozenset[int], available: tuple[int, ...],
              by_job: dict[int, ScheduledJob], entries: list[ScheduledJob]) -> None:
        nonlocal best_makespan, best_entries, nodes
        nodes += 1
        if not unscheduled:
            makespan = max(available)
            if makespan < best_makespan:
                best_makespan = makespan
                best_entries = list(entries)
            return
        if max(available) >= best_makespan:
            return
        ready = [j for j in unscheduled if all(p in by_job for p in jobs[j].predecessors)]
        for j in ready:
            predecessor_finish = max((by_job[p].finish for p in jobs[j].predecessors), default=0)
            seen_available: set[int] = set()
            for worker, free in enumerate(available):
                if free in seen_available:
                    continue
                seen_available.add(free)
                start = max(free, jobs[j].release, predecessor_finish)
                finish = start + jobs[j].duration
                if finish >= best_makespan:
                    continue
                entry = ScheduledJob(j, worker, start, finish)
                next_available = list(available)
                next_available[worker] = finish
                by_job[j] = entry
                visit(unscheduled - {j}, tuple(next_available), by_job, entries + [entry])
                del by_job[j]

    visit(frozenset(range(len(jobs))), tuple(0 for _ in range(workers)), {}, [])
    solution = ScheduleSolution(tuple(sorted(best_entries, key=lambda e: e.job)), best_makespan,
                                {"status": "optimal", "nodes_expanded": nodes})
    assert not validate_schedule(jobs, workers, solution)
    return solution


def time_indexed_ilp(jobs: Sequence[Job], workers: int,
                      time_limit_seconds: float) -> ScheduleSolution:
    """Solve an integer-time formulation using SciPy/HiGHS MILP."""
    validate_instance(jobs, workers)
    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    horizon = max(job.release for job in jobs) + sum(job.duration for job in jobs)
    starts: list[tuple[int, int, int]] = []
    by_job: list[list[int]] = [[] for _ in jobs]
    for j, job in enumerate(jobs):
        for worker in range(workers):
            for start in range(job.release, horizon - job.duration + 1):
                by_job[j].append(len(starts))
                starts.append((j, worker, start))
    cmax_index = len(starts)
    variable_count = cmax_index + 1
    c = np.zeros(variable_count)
    c[cmax_index] = 1.0
    integrality = np.ones(variable_count, dtype=int)
    integrality[cmax_index] = 0
    lower = np.zeros(variable_count)
    upper = np.ones(variable_count)
    upper[cmax_index] = horizon

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    lows: list[float] = []
    highs: list[float] = []

    def add_row(coefficients: Iterable[tuple[int, float]], low: float, high: float) -> None:
        row = len(lows)
        for column, value in coefficients:
            row_indices.append(row)
            col_indices.append(column)
            values.append(value)
        lows.append(low)
        highs.append(high)

    for j in range(len(jobs)):
        add_row(((index, 1.0) for index in by_job[j]), 1.0, 1.0)
    for successor, job in enumerate(jobs):
        for predecessor in job.predecessors:
            coefficients = [(index, float(start)) for index, (j, _, start) in enumerate(starts)
                            if j == successor]
            coefficients.extend((index, -float(start)) for index, (j, _, start) in enumerate(starts)
                                if j == predecessor)
            add_row(coefficients, float(jobs[predecessor].duration), np.inf)
    for worker in range(workers):
        for tick in range(horizon):
            active = [(index, 1.0) for index, (j, w, start) in enumerate(starts)
                      if w == worker and start <= tick < start + jobs[j].duration]
            if active:
                add_row(active, -np.inf, 1.0)
    for j, job in enumerate(jobs):
        coefficients = [(cmax_index, 1.0)]
        coefficients.extend((index, -float(start)) for index, (job_id, _, start) in enumerate(starts)
                            if job_id == j)
        add_row(coefficients, float(job.duration), np.inf)
    matrix = coo_matrix((values, (row_indices, col_indices)),
                        shape=(len(lows), variable_count)).tocsr()
    result = milp(c, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix, np.array(lows), np.array(highs)),
                  options={"time_limit": float(time_limit_seconds), "mip_rel_gap": 0.0,
                           "presolve": True})
    if result.x is None:
        raise RuntimeError(f"MILP returned no feasible incumbent: {result.message}")
    selected = [index for index in range(cmax_index) if result.x[index] > 0.5]
    entries = tuple(sorted(
        (ScheduledJob(j, worker, start, start + jobs[j].duration)
         for index in selected for j, worker, start in [starts[index]]),
        key=lambda entry: entry.job,
    ))
    makespan = max(entry.finish for entry in entries)
    status = "optimal" if result.status == 0 else "limit_reached_feasible"
    details: dict[str, object] = {
        "status": status,
        "scipy_status": int(result.status),
        "message": str(result.message),
        "horizon": horizon,
        "big_m": horizon,
        "binary_variables": cmax_index,
        "constraint_rows": len(lows),
        "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
        "mip_gap": float(getattr(result, "mip_gap", math.nan)),
        "dual_bound": float(getattr(result, "mip_dual_bound", math.nan)),
    }
    solution = ScheduleSolution(entries, makespan, details)
    violations = validate_schedule(jobs, workers, solution)
    if violations:
        raise RuntimeError(f"MILP incumbent failed validation: {violations}")
    return solution


def disjunctive_ilp(jobs: Sequence[Job], workers: int,
                    time_limit_seconds: float) -> ScheduleSolution:
    """Solve a compact integer-start, assignment and pair-order MILP.

    The serial horizon H=max(release)+sum(duration) bounds every completion.
    For two jobs assigned to the same worker, a binary order variable activates
    exactly one of two non-overlap inequalities. H is a sufficient big-M because
    every start is bounded by H-duration.
    """
    validate_instance(jobs, workers)
    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    count = len(jobs)
    horizon = max(job.release for job in jobs) + sum(job.duration for job in jobs)
    start_index = list(range(count))
    assignment_index = {(j, w): count + j * workers + w
                        for j in range(count) for w in range(workers)}
    next_index = count + count * workers
    order_index = {}
    for i in range(count):
        for j in range(i + 1, count):
            for worker in range(workers):
                order_index[(i, j, worker)] = next_index
                next_index += 1
    cmax_index = next_index
    variable_count = cmax_index + 1
    c = np.zeros(variable_count)
    c[cmax_index] = 1.0
    integrality = np.ones(variable_count, dtype=int)
    integrality[cmax_index] = 0
    lower = np.zeros(variable_count)
    upper = np.ones(variable_count)
    for j, job in enumerate(jobs):
        lower[start_index[j]] = job.release
        upper[start_index[j]] = horizon - job.duration
    upper[cmax_index] = horizon

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    lows: list[float] = []
    highs: list[float] = []

    def add_row(coefficients: Iterable[tuple[int, float]], low: float, high: float) -> None:
        row = len(lows)
        for column, value in coefficients:
            row_indices.append(row)
            col_indices.append(column)
            values.append(value)
        lows.append(low)
        highs.append(high)

    for j in range(count):
        add_row(((assignment_index[(j, w)], 1.0) for w in range(workers)), 1.0, 1.0)
    for successor, job in enumerate(jobs):
        for predecessor in job.predecessors:
            add_row(((start_index[successor], 1.0), (start_index[predecessor], -1.0)),
                    float(jobs[predecessor].duration), np.inf)
    for i in range(count):
        for j in range(i + 1, count):
            for worker in range(workers):
                order = order_index[(i, j, worker)]
                # If both assignments and order=1: j starts after i finishes.
                add_row(((start_index[j], 1.0), (start_index[i], -1.0),
                         (assignment_index[(i, worker)], -horizon),
                         (assignment_index[(j, worker)], -horizon),
                         (order, -horizon)),
                        float(jobs[i].duration - 3 * horizon), np.inf)
                # If both assignments and order=0: i starts after j finishes.
                add_row(((start_index[i], 1.0), (start_index[j], -1.0),
                         (assignment_index[(i, worker)], -horizon),
                         (assignment_index[(j, worker)], -horizon),
                         (order, horizon)),
                        float(jobs[j].duration - 2 * horizon), np.inf)
    for j, job in enumerate(jobs):
        add_row(((cmax_index, 1.0), (start_index[j], -1.0)),
                float(job.duration), np.inf)
    matrix = coo_matrix((values, (row_indices, col_indices)),
                        shape=(len(lows), variable_count)).tocsr()
    result = milp(c, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix, np.array(lows), np.array(highs)),
                  options={"time_limit": float(time_limit_seconds), "mip_rel_gap": 0.0,
                           "presolve": True})
    if result.x is None:
        raise RuntimeError(f"MILP returned no feasible incumbent: {result.message}")
    entries = []
    for j, job in enumerate(jobs):
        start = int(round(float(result.x[start_index[j]])))
        worker = max(range(workers), key=lambda w: result.x[assignment_index[(j, w)]])
        entries.append(ScheduledJob(j, worker, start, start + job.duration))
    makespan = max(entry.finish for entry in entries)
    status = "optimal" if result.status == 0 else "limit_reached_feasible"
    details: dict[str, object] = {
        "status": status, "scipy_status": int(result.status),
        "message": str(result.message), "formulation": "disjunctive_big_m",
        "horizon": horizon, "big_m": horizon,
        "integer_start_variables": count,
        "binary_variables": len(assignment_index) + len(order_index),
        "constraint_rows": len(lows),
        "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
        "mip_gap": float(getattr(result, "mip_gap", math.nan)),
        "dual_bound": float(getattr(result, "mip_dual_bound", math.nan)),
    }
    solution = ScheduleSolution(tuple(entries), makespan, details)
    violations = validate_schedule(jobs, workers, solution)
    if violations:
        raise RuntimeError(f"MILP incumbent failed validation: {violations}")
    return solution


def solution_to_dict(solution: ScheduleSolution) -> dict[str, object]:
    return {
        "makespan": solution.makespan,
        "entries": [entry.__dict__ for entry in solution.entries],
        "details": solution.details,
    }
