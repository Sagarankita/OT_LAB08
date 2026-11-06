import math

import pytest

from optimization_lab.scheduling import (
    Job, ScheduleSolution, ScheduledJob, disjunctive_ilp, exhaustive_schedule, generate_instance,
    list_schedule, schedule_lower_bound, time_indexed_ilp, validate_instance,
    validate_schedule,
)


def test_invalid_instance_is_rejected():
    with pytest.raises(ValueError):
        validate_instance([Job(0, 0)], 1)
    with pytest.raises(ValueError):
        validate_instance([Job(1, 0, (1,))], 1)


def test_validator_detects_release_precedence_overlap_and_makespan():
    jobs = [Job(2, 1), Job(2, 0, (0,))]
    bad = ScheduleSolution(
        (ScheduledJob(0, 0, 0, 2), ScheduledJob(1, 0, 1, 3)), 4, {}
    )
    violations = validate_schedule(jobs, 1, bad)
    assert any("release" in value for value in violations)
    assert any("precedence" in value for value in violations)
    assert any("overlaps" in value for value in violations)
    assert any("makespan" in value for value in violations)


def test_list_schedulers_are_feasible_and_deterministic():
    jobs = generate_instance(17, 10)
    for priority in ("fixed_order", "critical_path"):
        first = list_schedule(jobs, 3, priority)
        second = list_schedule(jobs, 3, priority)
        assert first == second
        assert validate_schedule(jobs, 3, first) == []
        assert first.makespan >= schedule_lower_bound(jobs, 3)


def test_exhaustive_and_ilp_match_known_small_optimum():
    jobs = [Job(3, 0), Job(2, 0), Job(4, 0, (0,)), Job(1, 1, (1,))]
    oracle = exhaustive_schedule(jobs, 2)
    ilp = time_indexed_ilp(jobs, 2, 5.0)
    assert oracle.makespan == ilp.makespan == 7
    assert ilp.details["status"] == "optimal"


def test_lower_bound_is_valid_against_exhaustive_oracle():
    for seed in range(5):
        jobs = generate_instance(seed, 7)
        oracle = exhaustive_schedule(jobs, 2)
        assert schedule_lower_bound(jobs, 2) <= oracle.makespan


def test_ilp_big_m_and_horizon_are_documented():
    jobs = generate_instance(8, 6)
    solution = disjunctive_ilp(jobs, 2, 5.0)
    expected = max(job.release for job in jobs) + sum(job.duration for job in jobs)
    assert solution.details["horizon"] == expected
    assert solution.details["big_m"] == expected
    assert math.isfinite(solution.details["dual_bound"])


def test_compact_ilp_matches_exhaustive_oracle():
    jobs = generate_instance(29, 7)
    assert disjunctive_ilp(jobs, 3, 5.0).makespan == exhaustive_schedule(jobs, 3).makespan
