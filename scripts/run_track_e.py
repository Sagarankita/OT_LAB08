"""Run the Track E precedence-constrained job-scheduling case study."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_lab.benchmarking import (
    BenchmarkConfig, RunRecorder, canonical_json_bytes, numeric_summary,
    record_invalid_attempt,
)
from optimization_lab.scheduling import (
    Job, disjunctive_ilp, exhaustive_schedule, generate_instance, list_schedule,
    schedule_lower_bound, solution_to_dict, validate_schedule,
)

METHODS = ("fixed_order", "critical_path", "disjunctive_ilp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def solve(method: str, jobs: list[Job], workers: int, ilp_limit: float):
    if method == "fixed_order":
        return list_schedule(jobs, workers, "fixed_order")
    if method == "critical_path":
        return list_schedule(jobs, workers, "critical_path")
    if method == "disjunctive_ilp":
        return disjunctive_ilp(jobs, workers, ilp_limit)
    raise ValueError(f"unknown method {method}")


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
               "--results-root", str(args.results_root.resolve())]
    try:
        raw_bytes = config_path.read_bytes()
        config = BenchmarkConfig.from_mapping(json.loads(raw_bytes))
        settings = config.solver_settings
        if settings.get("methods") != list(METHODS):
            raise ValueError(f"methods must be {list(METHODS)} in that order")
        workers = int(settings["workers"])
        sizes = [int(value) for value in settings["scaling_sizes"]]
        small_size = int(settings["small_size"])
        if not sizes or sizes[0] != small_size or sizes != sorted(set(sizes)):
            raise ValueError("scaling_sizes must be sorted, unique and start with small_size")
        if small_size > 9 or max(sizes) != config.input_size:
            raise ValueError("small_size must be <=9 and input_size must equal largest scale")
        if workers <= 0:
            raise ValueError("workers must be positive")
        ilp_limit = float(settings["ilp_time_limit_seconds"])
        if ilp_limit <= 0:
            raise ValueError("ILP time limit must be positive")
    except Exception as exc:
        raw_bytes = config_path.read_bytes() if config_path.is_file() else b""
        directory = record_invalid_attempt(args.results_root.resolve(), config_path,
                                           raw_bytes, exc, command)
        print(f"INVALID: {exc}\nEvidence: {directory}", file=sys.stderr)
        return 2

    recorder = RunRecorder(
        args.results_root.resolve(), config_path, config,
        source_files=[Path(__file__).resolve(), ROOT / "src/optimization_lab/scheduling.py",
                      ROOT / "src/optimization_lab/benchmarking.py"], command=command,
    )
    total_started = time.perf_counter()
    try:
        instances: dict[tuple[int, int], list[Job]] = {}
        instance_artifact: list[dict[str, object]] = []
        setup_started = time.perf_counter()
        for seed in config.seeds:
            for size in sizes:
                generation_seed = seed * 10_000 + size
                jobs = generate_instance(
                    generation_seed, size, float(settings["edge_probability"]),
                    int(settings["max_predecessors"]),
                    tuple(int(x) for x in settings["duration_range"]),
                    tuple(int(x) for x in settings["release_range"]),
                )
                instances[(seed, size)] = jobs
                lower_bound = schedule_lower_bound(jobs, workers)
                oracle = exhaustive_schedule(jobs, workers) if size == small_size else None
                instance_artifact.append({
                    "seed": seed, "generation_seed": generation_seed, "job_count": size,
                    "workers": workers, "lower_bound": lower_bound,
                    "jobs": [{"duration": job.duration, "release": job.release,
                              "predecessors": list(job.predecessors)} for job in jobs],
                    "exhaustive_oracle": solution_to_dict(oracle) if oracle else None,
                })
        oracle_setup_seconds = time.perf_counter() - setup_started

        warmup_started = time.perf_counter()
        for _ in range(config.warmup_repetitions):
            for jobs in instances.values():
                for method in METHODS:
                    solve(method, jobs, workers, ilp_limit)
        warmup_seconds = time.perf_counter() - warmup_started

        deadline = total_started + config.timeout_seconds
        for seed in config.seeds:
            for size in sizes:
                jobs = instances[(seed, size)]
                lower_bound = schedule_lower_bound(jobs, workers)
                oracle = next(
                    item["exhaustive_oracle"] for item in instance_artifact
                    if item["seed"] == seed and item["job_count"] == size
                )
                oracle_value = oracle["makespan"] if oracle else None
                for repetition in range(config.repetitions):
                    repetition_solutions = {}
                    timings = {}
                    for method in METHODS:
                        if time.perf_counter() > deadline:
                            raise TimeoutError(f"Track E exceeded {config.timeout_seconds} seconds")
                        started = time.perf_counter()
                        solution = solve(method, jobs, workers, ilp_limit)
                        elapsed = time.perf_counter() - started
                        violations = validate_schedule(jobs, workers, solution)
                        if violations:
                            raise AssertionError(f"{method} infeasible: {violations}")
                        repetition_solutions[method] = solution
                        timings[method] = elapsed
                    baseline = repetition_solutions["fixed_order"].makespan
                    for method in METHODS:
                        solution = repetition_solutions[method]
                        objective_gap = (None if oracle_value is None else
                                         100.0 * (solution.makespan - oracle_value) / oracle_value)
                        bound_gap = 100.0 * (solution.makespan - lower_bound) / solution.makespan
                        details = solution.details
                        recorder.add_record({
                            "seed": seed, "job_count": size, "workers": workers,
                            "repetition": repetition, "method": method,
                            "makespan": solution.makespan, "feasible": True,
                            "lower_bound": lower_bound,
                            "lower_bound_gap_percent": bound_gap,
                            "oracle_makespan": oracle_value,
                            "optimality_gap_percent": objective_gap,
                            "baseline_improvement_percent":
                                100.0 * (baseline - solution.makespan) / baseline,
                            "execution_seconds": timings[method],
                            "solver_status": details.get("status", "not_applicable"),
                            "solver_dual_bound": details.get("dual_bound", None),
                            "solver_mip_gap_percent": (
                                100.0 * float(details["mip_gap"])
                                if method == "disjunctive_ilp" and
                                math.isfinite(float(details.get("mip_gap", math.nan))) else None
                            ),
                            "solver_nodes": details.get("mip_node_count", None),
                            "horizon": details.get("horizon", None),
                            "binary_variables": details.get("binary_variables", None),
                            "schedule": solution_to_dict(solution)["entries"],
                        })

        instances_path = recorder.directory / "instances.json"
        write_json(instances_path, instance_artifact)
        timing = numeric_summary(recorder.records, "method", "execution_seconds")
        makespan = numeric_summary(recorder.records, "method", "makespan")
        improvements = numeric_summary(recorder.records, "method", "baseline_improvement_percent")
        by_size: dict[str, dict[str, dict[str, float]]] = {}
        for size in sizes:
            size_records = [record for record in recorder.records if record["job_count"] == size]
            by_size[str(size)] = {
                "makespan": numeric_summary(size_records, "method", "makespan"),
                "execution_seconds": numeric_summary(size_records, "method", "execution_seconds"),
                "baseline_improvement_percent": numeric_summary(
                    size_records, "method", "baseline_improvement_percent"
                ),
            }

        comparison_plot = recorder.directory / "scheduling_comparison.png"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for method in METHODS:
            axes[0].plot(sizes, [by_size[str(size)]["makespan"][method]["mean"]
                                 for size in sizes], marker="o", label=method)
            axes[1].plot(sizes, [by_size[str(size)]["execution_seconds"][method]["mean"]
                                 for size in sizes], marker="o", label=method)
        axes[0].set(xlabel="Jobs", ylabel="Mean makespan (time units)",
                    title="Schedule quality (lower is better)")
        axes[1].set(xlabel="Jobs", ylabel="Mean execution time (seconds, log scale)",
                    title="Computation cost")
        axes[1].set_yscale("log")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        fig.suptitle("Track E — identical DAG instances, three workers")
        fig.tight_layout()
        fig.savefig(comparison_plot, dpi=140)
        plt.close(fig)

        status_counts: dict[str, int] = defaultdict(int)
        for record in recorder.records:
            if record["method"] == "disjunctive_ilp":
                status_counts[str(record["solver_status"])] += 1
        recorder.metadata["timing_policy"] = {
            "clock": "time.perf_counter", "generation_and_oracles_excluded": True,
            "oracle_setup_seconds": oracle_setup_seconds,
            "warmups_excluded": True, "warmup_seconds": warmup_seconds,
            "warmup_repetitions": config.warmup_repetitions,
            "measured_repetitions": config.repetitions,
            "MILP_time_includes_model_construction": True,
        }
        recorder.metadata["formulation"] = {
            "time_discretization": "integer start times; durations and releases are integers",
            "horizon_and_big_m": "max release + sum durations; a valid serial upper bound",
            "decision_variables": (
                "integer starts, binary worker assignments and pair-order variables, "
                "plus continuous C_max"
            ),
            "objective": "minimize C_max",
        }
        write_json(recorder.directory / "metadata.json", recorder.metadata)
        summary = {
            "status": "completed", "scope": "Track E main job-scheduling case study",
            "baseline": "fixed-order ready-list scheduler", "methods": list(METHODS),
            "seeds": list(config.seeds), "sizes": sizes, "workers": workers,
            "record_count": len(recorder.records), "timing_by_method": timing,
            "makespan_by_method": makespan,
            "baseline_improvement_percent_by_method": improvements,
            "by_size": by_size, "ilp_status_counts": dict(status_counts),
            "small_oracle_instances": len(config.seeds),
            "setup_seconds": oracle_setup_seconds, "warmup_seconds": warmup_seconds,
        }
        recorder.complete(summary, {
            "instances": instances_path, "comparison_plot": comparison_plot,
        })
        print(recorder.directory)
        return 0
    except Exception as exc:
        recorder.fail(exc)
        print(f"FAILED: {exc}\nEvidence: {recorder.directory}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
