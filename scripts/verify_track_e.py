"""Independently validate Track E schedules, objectives, bounds and small optima."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.benchmarking import load_completed_run, sha256_file


def independent_validate(jobs, workers, record):
    entries = record["schedule"]
    if len(entries) != len(jobs) or {e["job"] for e in entries} != set(range(len(jobs))):
        return False
    by_job = {entry["job"]: entry for entry in entries}
    for j, job in enumerate(jobs):
        entry = by_job[j]
        if not 0 <= entry["worker"] < workers:
            return False
        if entry["start"] < job["release"]:
            return False
        if entry["finish"] != entry["start"] + job["duration"]:
            return False
        if any(entry["start"] < by_job[p]["finish"] for p in job["predecessors"]):
            return False
    for worker in range(workers):
        assigned = sorted((entry for entry in entries if entry["worker"] == worker),
                          key=lambda entry: (entry["start"], entry["job"]))
        if any(right["start"] < left["finish"] for left, right in zip(assigned, assigned[1:])):
            return False
    return record["makespan"] == max(entry["finish"] for entry in entries)


def critical_lengths(jobs):
    successors = [[] for _ in jobs]
    for j, job in enumerate(jobs):
        for predecessor in job["predecessors"]:
            successors[predecessor].append(j)
    lengths = [0] * len(jobs)
    for j in range(len(jobs) - 1, -1, -1):
        lengths[j] = jobs[j]["duration"] + max((lengths[s] for s in successors[j]), default=0)
    return lengths


def independent_lower_bound(jobs, workers):
    work = math.ceil(sum(job["duration"] for job in jobs) / workers)
    critical = critical_lengths(jobs)
    return max(work, max(job["release"] + critical[j] for j, job in enumerate(jobs)))


def independent_exhaustive_makespan(jobs, workers):
    """Tiny independent enumeration; returns only the optimal objective."""
    incumbent = sum(job["duration"] for job in jobs) + max(job["release"] for job in jobs)
    best = incumbent

    def visit(remaining, available, finish_times):
        nonlocal best
        if not remaining:
            best = min(best, max(available))
            return
        if max(available) >= best:
            return
        ready = [j for j in remaining if all(p in finish_times for p in jobs[j]["predecessors"])]
        for j in ready:
            predecessor_finish = max((finish_times[p] for p in jobs[j]["predecessors"]), default=0)
            seen = set()
            for worker, free in enumerate(available):
                if free in seen:
                    continue
                seen.add(free)
                start = max(free, jobs[j]["release"], predecessor_finish)
                finish = start + jobs[j]["duration"]
                if finish >= best:
                    continue
                next_available = list(available)
                next_available[worker] = finish
                finish_times[j] = finish
                visit(remaining - {j}, tuple(next_available), finish_times)
                del finish_times[j]

    visit(set(range(len(jobs))), tuple(0 for _ in range(workers)), {})
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    loaded = load_completed_run(run)
    config, metadata, records, summary = (
        loaded["config"], loaded["metadata"], loaded["records"], loaded["summary"]
    )
    instances = json.loads((run / "instances.json").read_text())
    lookup = {(item["seed"], item["job_count"]): item for item in instances}
    methods = set(config["solver_settings"]["methods"])
    expected = len(config["seeds"]) * len(config["solver_settings"]["scaling_sizes"]) * \
        config["repetitions"] * len(methods)
    schedules_valid = all(independent_validate(
        lookup[(record["seed"], record["job_count"])]["jobs"],
        record["workers"], record,
    ) for record in records)
    lower_bounds = {
        key: independent_lower_bound(instance["jobs"], instance["workers"])
        for key, instance in lookup.items()
    }
    bounds_correct = all(
        record["lower_bound"] == lower_bounds[(record["seed"], record["job_count"])]
        and record["lower_bound"] <= record["makespan"] for record in records
    )
    gaps_correct = all(math.isclose(
        record["lower_bound_gap_percent"],
        100.0 * (record["makespan"] - record["lower_bound"]) / record["makespan"],
        abs_tol=1e-10,
    ) for record in records)
    small_size = config["solver_settings"]["small_size"]
    independent_optima = {
        seed: independent_exhaustive_makespan(lookup[(seed, small_size)]["jobs"],
                                              lookup[(seed, small_size)]["workers"])
        for seed in config["seeds"]
    }
    oracle_correct = all(
        lookup[(seed, small_size)]["exhaustive_oracle"]["makespan"] == independent_optima[seed]
        for seed in config["seeds"]
    )
    small_ilp_optimal = all(
        record["makespan"] == independent_optima[record["seed"]] and
        record["solver_status"] == "optimal"
        for record in records
        if record["job_count"] == small_size and record["method"] == "disjunctive_ilp"
    )
    oracle_gaps_correct = all(
        record["optimality_gap_percent"] is None or math.isclose(
            record["optimality_gap_percent"],
            100.0 * (record["makespan"] - independent_optima[record["seed"]]) /
            independent_optima[record["seed"]], abs_tol=1e-10,
        ) for record in records
    )
    paired = all(
        {record["method"] for record in records
         if record["seed"] == seed and record["job_count"] == size and
         record["repetition"] == repetition} == methods
        for seed in config["seeds"]
        for size in config["solver_settings"]["scaling_sizes"]
        for repetition in range(config["repetitions"])
    )
    baselines = {(record["seed"], record["job_count"], record["repetition"]): record["makespan"]
                 for record in records if record["method"] == "fixed_order"}
    improvements_correct = all(math.isclose(
        record["baseline_improvement_percent"],
        100.0 * (baselines[(record["seed"], record["job_count"], record["repetition"])] -
                 record["makespan"]) /
        baselines[(record["seed"], record["job_count"], record["repetition"])],
        abs_tol=1e-10,
    ) for record in records)
    ilp_status_honest = all(
        (record["solver_status"] == "optimal" and record["solver_mip_gap_percent"] <= 1e-7) or
        (record["solver_status"] == "limit_reached_feasible" and
         record["solver_mip_gap_percent"] is not None)
        for record in records if record["method"] == "disjunctive_ilp"
    )
    status = json.loads((run / "status.json").read_text())
    hashes_match = all(
        sha256_file(run / artifact["path"]) == artifact["sha256"]
        for artifact in status["artifacts"].values()
    )
    checks = {
        "final_configuration": config["run_kind"] == "final",
        "five_seed_instances": len(config["seeds"]) == 5,
        "three_scales_including_small_and_larger":
            len(config["solver_settings"]["scaling_sizes"]) >= 3,
        "record_count": len(records) == expected == summary["record_count"],
        "all_method_pairs_present": paired,
        "all_schedules_independently_valid": schedules_valid,
        "lower_bounds_independently_recomputed": bounds_correct,
        "lower_bound_gaps_recomputed": gaps_correct,
        "small_exhaustive_oracles_independently_recomputed": oracle_correct,
        "small_ilp_matches_optimum_and_proves_optimal": small_ilp_optimal,
        "small_optimality_gaps_recomputed": oracle_gaps_correct,
        "baseline_improvements_recomputed": improvements_correct,
        "ilp_status_and_gap_preserved": ilp_status_honest,
        "integer_time_and_big_m_documented":
            "horizon_and_big_m" in metadata["formulation"],
        "timing_policy_recorded": metadata["timing_policy"]["warmups_excluded"] is True,
        "all_declared_artifact_hashes_match": hashes_match,
        "git_provenance_pending_user":
            metadata["provenance"]["git_revision"] == "pending user Git work",
    }
    report = {
        "checks": checks, "passed": all(checks.values()),
        "raw_records": len(records), "instances": len(instances),
        "small_optima": independent_optima,
    }
    (run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
