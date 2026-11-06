#!/usr/bin/env python3
"""Run Track D submodular and real parallel-computing experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.parallel_optimization import (  # noqa: E402
    Task,
    exhaustive_max_coverage,
    greedy_max_coverage,
    run_data_parallel,
    run_model_parallel,
    run_pipeline,
    run_tasks,
    verify_submodular_axioms,
)


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_sets(seed: int, count: int, universe: int) -> list[frozenset[int]]:
    rng = np.random.default_rng(seed)
    return [
        frozenset(int(value) for value in rng.choice(universe, size=int(rng.integers(4, 12)), replace=False))
        for _ in range(count)
    ]


def make_tasks(count: int, scale: float, seed: int) -> list[Task]:
    base = [500_000, 70_000, 130_000, 230_000]
    return [
        Task(
            task_id=index,
            work=max(100, int(scale * (base[index % 4] + (index % 5) * 9_000))),
            block=index % 4,
            seed=seed,
        )
        for index in range(count)
    ]


def summarize(samples: list[float]) -> dict:
    return {
        "samples_seconds": samples,
        "mean_seconds": statistics.mean(samples),
        "stdev_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def run_repeated(function, warmups: int, repetitions: int) -> tuple[list[dict], list[dict]]:
    warmup_results = [function() for _ in range(warmups)]
    measured = [function() for _ in range(repetitions)]
    return warmup_results, measured


def hash_outputs(outputs: object) -> str:
    return hashlib.sha256(json.dumps(outputs, sort_keys=True).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if max(config["worker_counts"]) > (os.cpu_count() or 1):
        raise ValueError("worker count exceeds available logical CPUs")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = ROOT / "results" / "track_d_parallel" / run_id
    run_dir.mkdir(parents=True)
    dump(run_dir / "config.json", config)
    records: list[dict] = []

    axiom_sets = make_sets(config["seed"], 7, 24)
    axiom_checks = verify_submodular_axioms(axiom_sets)
    if not all(axiom_checks[name] for name in ("normalized", "nonnegative", "monotone", "diminishing_returns")):
        raise AssertionError("maximum coverage failed a submodular axiom")
    submodular_results = []
    for offset in range(config["submodular_instances"]):
        sets = make_sets(config["seed"] + offset + 1, 12, 40)
        greedy_selected, greedy_value = greedy_max_coverage(sets, 4)
        exact_selected, exact_value = exhaustive_max_coverage(sets, 4)
        result = {
            "seed": config["seed"] + offset + 1,
            "greedy_selected": greedy_selected,
            "greedy_value": greedy_value,
            "exact_selected": exact_selected,
            "exact_value": exact_value,
            "ratio": greedy_value / exact_value,
        }
        submodular_results.append(result)
        records.append({"experiment": "submodular", "method": "greedy", **result})
    dump(run_dir / "submodular.json", {"axioms": axiom_checks, "instances": submodular_results})

    tasks = make_tasks(config["fixed_task_count"], config["work_scale"], config["seed"])
    baseline = run_tasks(tasks, "serial", 1)
    expected_hash = hash_outputs(baseline["outputs"])
    scheduling_runs: dict[str, dict[str, list[dict]]] = {}
    methods = ["serial", "static", "dynamic", "work_stealing"]
    for method in methods:
        worker_values = [1] if method == "serial" else config["worker_counts"]
        scheduling_runs[method] = {}
        for workers in worker_values:
            warm, measured = run_repeated(
                lambda m=method, w=workers: run_tasks(tasks, m, w),
                config["warmups"],
                config["repetitions"],
            )
            if any(hash_outputs(result["outputs"]) != expected_hash for result in warm + measured):
                raise AssertionError(f"{method}/{workers} output mismatch")
            scheduling_runs[method][str(workers)] = measured
            for repetition, result in enumerate(measured):
                records.append(
                    {
                        "experiment": "fixed_scheduling",
                        "method": method,
                        "workers": workers,
                        "repetition": repetition,
                        "elapsed_seconds": result["elapsed_seconds"],
                        "residual_overhead_seconds": result["residual_overhead_seconds"],
                        "steals": result["steals"],
                        "remote_tasks": result["remote_tasks"],
                        "checksum": result["checksum"],
                    }
                )

    fixed_serial_mean = statistics.mean(
        result["elapsed_seconds"] for result in scheduling_runs["serial"]["1"]
    )
    scheduling_summary = {}
    for method, by_worker in scheduling_runs.items():
        scheduling_summary[method] = {}
        for workers_text, runs in by_worker.items():
            workers = int(workers_text)
            timing = summarize([result["elapsed_seconds"] for result in runs])
            speedup = fixed_serial_mean / timing["mean_seconds"]
            timing.update(
                {
                    "speedup": speedup,
                    "efficiency": speedup / workers,
                    "mean_residual_overhead_seconds": statistics.mean(
                        result["residual_overhead_seconds"] for result in runs
                    ),
                    "mean_steals": statistics.mean(result["steals"] for result in runs),
                    "mean_remote_tasks": statistics.mean(result["remote_tasks"] for result in runs),
                }
            )
            scheduling_summary[method][workers_text] = timing

    scaled_results = {}
    for workers in config["worker_counts"]:
        scaled_tasks = make_tasks(
            config["scaled_tasks_per_worker"] * workers,
            config["work_scale"],
            config["seed"] + workers,
        )
        serial_warm, serial_runs = run_repeated(
            lambda t=scaled_tasks: run_tasks(t, "serial", 1),
            config["warmups"],
            config["repetitions"],
        )
        steal_warm, steal_runs = run_repeated(
            lambda t=scaled_tasks, w=workers: run_tasks(t, "work_stealing", w),
            config["warmups"],
            config["repetitions"],
        )
        serial_hash = hash_outputs(serial_runs[0]["outputs"])
        if any(hash_outputs(run["outputs"]) != serial_hash for run in serial_warm + steal_warm + steal_runs):
            raise AssertionError("scaled work-stealing output mismatch")
        serial_mean = statistics.mean(run["elapsed_seconds"] for run in serial_runs)
        parallel_mean = statistics.mean(run["elapsed_seconds"] for run in steal_runs)
        scaled_results[str(workers)] = {
            "tasks": len(scaled_tasks),
            "serial": summarize([run["elapsed_seconds"] for run in serial_runs]),
            "work_stealing": summarize([run["elapsed_seconds"] for run in steal_runs]),
            "scaled_speedup": serial_mean / parallel_mean,
            "scaled_efficiency": serial_mean / parallel_mean / workers,
        }

    data_values = list(range(config["data_size"]))
    data_results = {}
    for workers in config["worker_counts"]:
        _, measured = run_repeated(
            lambda w=workers: run_data_parallel(data_values, w),
            config["warmups"],
            config["repetitions"],
        )
        data_results[str(workers)] = measured
    if len({run["result"] for runs in data_results.values() for run in runs}) != 1:
        raise AssertionError("data-parallel results differ")

    rng = np.random.default_rng(config["seed"])
    features = rng.normal(size=(config["model_rows"], config["model_features"]))
    weights = rng.normal(size=config["model_features"])
    model_results = {}
    model_reference = None
    for workers in config["worker_counts"]:
        _, measured = run_repeated(
            lambda w=workers: run_model_parallel(features, weights, w),
            config["warmups"],
            config["repetitions"],
        )
        if model_reference is None:
            model_reference = measured[0]["prediction"]
        if any(not np.allclose(result["prediction"], model_reference, rtol=1e-10, atol=1e-10) for result in measured):
            raise AssertionError("model-parallel predictions differ")
        model_results[str(workers)] = [
            {key: value for key, value in result.items() if key != "prediction"} for result in measured
        ]

    batches = [
        list(range(index * config["pipeline_batch_size"], (index + 1) * config["pipeline_batch_size"]))
        for index in range(config["pipeline_batches"])
    ]
    pipeline_results = {}
    for parallel in (False, True):
        _, measured = run_repeated(
            lambda p=parallel: run_pipeline(batches, p),
            config["warmups"],
            config["repetitions"],
        )
        pipeline_results["parallel" if parallel else "serial"] = measured
    pipeline_hashes = {
        hash_outputs(run["outputs"]) for runs in pipeline_results.values() for run in runs
    }
    if len(pipeline_hashes) != 1:
        raise AssertionError("pipeline outputs differ")

    paradigms = {}
    for name, result_map in (("data", data_results), ("model", model_results)):
        serial_mean = statistics.mean(run["elapsed_seconds"] for run in result_map["1"])
        paradigms[name] = {}
        for workers_text, runs in result_map.items():
            timing = summarize([run["elapsed_seconds"] for run in runs])
            workers = int(workers_text)
            timing["speedup"] = serial_mean / timing["mean_seconds"]
            timing["efficiency"] = timing["speedup"] / workers
            timing["communicated_bytes"] = runs[0]["communicated_bytes"]
            paradigms[name][workers_text] = timing
    pipeline_serial = statistics.mean(run["elapsed_seconds"] for run in pipeline_results["serial"])
    paradigms["pipeline"] = {
        name: {
            **summarize([run["elapsed_seconds"] for run in runs]),
            "speedup": pipeline_serial / statistics.mean(run["elapsed_seconds"] for run in runs),
            "communicated_bytes": runs[0]["communicated_bytes"],
        }
        for name, runs in pipeline_results.items()
    }

    amdahl = {}
    for method in ("static", "dynamic", "work_stealing"):
        amdahl[method] = {}
        for workers_text, timing in scheduling_summary[method].items():
            workers = int(workers_text)
            speedup = timing["speedup"]
            serial_fraction = None if workers == 1 else (1.0 / speedup - 1.0 / workers) / (1.0 - 1.0 / workers)
            amdahl[method][workers_text] = {
                "fitted_serial_fraction": serial_fraction,
                "model_valid_fraction": serial_fraction is not None and 0.0 <= serial_fraction <= 1.0,
                "observed_speedup": speedup,
            }

    for workers_text, result in scaled_results.items():
        workers = int(workers_text)
        if workers == 1:
            result["gustafson_predicted_scaled_speedup"] = 1.0
            result["gustafson_serial_fraction_source"] = "identity at one worker"
        else:
            serial_fraction = amdahl["work_stealing"][workers_text]["fitted_serial_fraction"]
            result["gustafson_predicted_scaled_speedup"] = (
                workers - serial_fraction * (workers - 1)
                if serial_fraction is not None and 0.0 <= serial_fraction <= 1.0
                else None
            )
            result["gustafson_serial_fraction_source"] = "fixed-size Amdahl fit; null when model assumptions fail"

    raw_parallel = {
        "fixed": scheduling_runs,
        "scaled": scaled_results,
        "data": data_results,
        "model": model_results,
        "pipeline": pipeline_results,
    }
    dump(run_dir / "parallel_raw.json", raw_parallel)
    dump(run_dir / "records.json", records)
    if records:
        fields = sorted({key for record in records for key in record})
        with (run_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    summary = {
        "run_kind": config["run_kind"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "multiprocessing_start_method": "spawn",
            "blas_threads_requested": 1,
        },
        "submodular": {
            "axioms": axiom_checks,
            "instances": len(submodular_results),
            "minimum_greedy_ratio": min(result["ratio"] for result in submodular_results),
            "mean_greedy_ratio": statistics.mean(result["ratio"] for result in submodular_results),
            "guarantee_threshold": 1.0 - 1.0 / math.e,
            "guarantee_conditions": "normalized, nonnegative, monotone submodular objective; cardinality budget; exact marginal-gain greedy",
        },
        "fixed_scheduling": scheduling_summary,
        "scaled_work_stealing": scaled_results,
        "parallel_paradigms": paradigms,
        "amdahl_fits": amdahl,
        "correctness": {
            "fixed_output_sha256": expected_hash,
            "all_fixed_outputs_equal": True,
            "all_scaled_outputs_equal": True,
            "data_outputs_equal": True,
            "model_outputs_allclose": True,
            "pipeline_outputs_equal": True,
        },
    }
    dump(run_dir / "summary.json", summary)

    worker_counts = config["worker_counts"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for method, marker in (("static", "o"), ("dynamic", "s"), ("work_stealing", "^")):
        axes[0].plot(
            worker_counts,
            [scheduling_summary[method][str(worker)]["speedup"] for worker in worker_counts],
            marker=marker,
            label=method.replace("_", " "),
        )
    axes[0].plot(worker_counts, worker_counts, "--", color="gray", label="ideal")
    axes[0].set_xlabel("Workers")
    axes[0].set_ylabel("Fixed-size speedup")
    axes[0].set_title("Scheduling speedup")
    axes[0].legend()
    names = ["data", "model", "pipeline"]
    values = [
        paradigms["data"][str(max(worker_counts))]["speedup"],
        paradigms["model"][str(max(worker_counts))]["speedup"],
        paradigms["pipeline"]["parallel"]["speedup"],
    ]
    axes[1].bar(names, values, color=["#60a5fa", "#34d399", "#f59e0b"])
    axes[1].axhline(1.0, color="gray", linestyle="--")
    axes[1].set_ylabel("Speedup vs serial")
    axes[1].set_title("Parallelism overhead can dominate")
    fig.tight_layout()
    fig.savefig(run_dir / "parallel_comparison.png", dpi=160)
    plt.close(fig)

    dump(run_dir / "status.json", {"passed": True, "run_directory": run_dir.relative_to(ROOT).as_posix()})
    dump(
        ROOT / "results" / "track_d_parallel" / "latest-success.json",
        {
            "passed": True,
            "run_kind": config["run_kind"],
            "run_directory": run_dir.relative_to(ROOT).as_posix(),
            "records": len(records),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
