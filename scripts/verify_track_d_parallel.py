#!/usr/bin/env python3
"""Independently verify Track D submodular/parallel artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.parallel_optimization import (  # noqa: E402
    Task,
    coverage_value,
    exhaustive_max_coverage,
    run_data_parallel,
    run_model_parallel,
    run_pipeline,
    run_tasks,
)


def load(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def make_sets(seed: int, count: int, universe: int) -> list[frozenset[int]]:
    rng = np.random.default_rng(seed)
    return [
        frozenset(int(value) for value in rng.choice(universe, size=int(rng.integers(4, 12)), replace=False))
        for _ in range(count)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run_dir = args.run_directory.resolve()
    config = load(run_dir / "config.json")
    summary = load(run_dir / "summary.json")
    raw = load(run_dir / "parallel_raw.json")
    submodular = load(run_dir / "submodular.json")
    records = load(run_dir / "records.json")

    exact_reproduced = True
    greedy_values_valid = True
    for item in submodular["instances"]:
        sets = make_sets(item["seed"], 12, 40)
        _, exact = exhaustive_max_coverage(sets, 4)
        exact_reproduced &= exact == item["exact_value"]
        greedy_values_valid &= coverage_value(item["greedy_selected"], sets) == item["greedy_value"]

    fixed_serial = summary["fixed_scheduling"]["serial"]["1"]["mean_seconds"]
    speedups_valid = True
    efficiencies_valid = True
    for method in ("static", "dynamic", "work_stealing"):
        for workers_text, item in summary["fixed_scheduling"][method].items():
            workers = int(workers_text)
            speedups_valid &= math.isclose(item["speedup"], fixed_serial / item["mean_seconds"], rel_tol=1e-12)
            efficiencies_valid &= math.isclose(item["efficiency"], item["speedup"] / workers, rel_tol=1e-12)

    # Fresh small process-based checks, independent of the saved correctness flags.
    tasks = [Task(index, 2_000 + 100 * index, index % 2, 91) for index in range(8)]
    serial = run_tasks(tasks, "serial", 1)
    fresh_schedulers_equal = all(
        run_tasks(tasks, method, 2)["outputs"] == serial["outputs"]
        for method in ("static", "dynamic", "work_stealing")
    )
    values = list(range(2_000))
    fresh_data_equal = run_data_parallel(values, 1)["result"] == run_data_parallel(values, 2)["result"]
    rng = np.random.default_rng(99)
    features = rng.normal(size=(64, 8))
    weights = rng.normal(size=8)
    fresh_model_equal = np.allclose(
        run_model_parallel(features, weights, 1)["prediction"],
        run_model_parallel(features, weights, 2)["prediction"],
        rtol=1e-12,
        atol=1e-12,
    )
    batches = [list(range(index * 20, index * 20 + 20)) for index in range(4)]
    fresh_pipeline_equal = run_pipeline(batches, False)["outputs"] == run_pipeline(batches, True)["outputs"]

    all_fixed_hashes_equal = True
    expected = summary["correctness"]["fixed_output_sha256"]
    import hashlib
    for by_worker in raw["fixed"].values():
        for runs in by_worker.values():
            for run in runs:
                digest = hashlib.sha256(json.dumps(run["outputs"], sort_keys=True).encode("utf-8")).hexdigest()
                all_fixed_hashes_equal &= digest == expected

    axioms = submodular["axioms"]
    checks = {
        "final_configuration": config["run_kind"] == "final",
        "worker_counts_within_hardware": max(config["worker_counts"]) <= (os.cpu_count() or 1),
        "multiple_worker_counts": len(config["worker_counts"]) >= 3,
        "submodular_normalized": axioms["normalized"] is True,
        "submodular_nonnegative": axioms["nonnegative"] is True,
        "submodular_monotone": axioms["monotone"] is True,
        "submodular_diminishing_returns": axioms["diminishing_returns"] is True,
        "exhaustive_property_counts": axioms["subsets_checked"] == 128 and axioms["diminishing_comparisons"] > 0,
        "exact_optima_reproduced": exact_reproduced,
        "greedy_values_reproduced": greedy_values_valid,
        "greedy_above_guarantee": min(item["ratio"] for item in submodular["instances"]) >= 1 - 1 / math.e,
        "fixed_outputs_hash_equal": all_fixed_hashes_equal,
        "saved_correctness_flags": all(summary["correctness"].values()),
        "speedups_recomputed": speedups_valid,
        "efficiencies_recomputed": efficiencies_valid,
        "three_repetitions": all(
            len(item["samples_seconds"]) == 3
            for method in summary["fixed_scheduling"].values()
            for item in method.values()
        ),
        "real_steals_observed": summary["fixed_scheduling"]["work_stealing"]["4"]["mean_steals"] > 0,
        "locality_violations_recorded": summary["fixed_scheduling"]["work_stealing"]["4"]["mean_remote_tasks"] > 0,
        "residual_overhead_recorded": all(
            item["mean_residual_overhead_seconds"] >= 0
            for method in summary["fixed_scheduling"].values()
            for item in method.values()
        ),
        "fixed_size_speedup_observed": max(
            summary["fixed_scheduling"][method]["4"]["speedup"]
            for method in ("static", "dynamic", "work_stealing")
        ) > 1,
        "scaled_experiments_present": set(summary["scaled_work_stealing"]) == {"1", "2", "4"},
        "amdahl_fits_label_validity": all(
            item["model_valid_fraction"] == (item["fitted_serial_fraction"] is not None and 0 <= item["fitted_serial_fraction"] <= 1)
            for method in summary["amdahl_fits"].values()
            for item in method.values()
        ),
        "gustafson_predictions_present": all(
            "gustafson_predicted_scaled_speedup" in item for item in summary["scaled_work_stealing"].values()
        ),
        "communication_bytes_recorded": all(
            summary["parallel_paradigms"][name]["4"]["communicated_bytes"] > 0
            for name in ("data", "model")
        ) and summary["parallel_paradigms"]["pipeline"]["parallel"]["communicated_bytes"] > 0,
        "fresh_scheduler_outputs_equal": fresh_schedulers_equal,
        "fresh_data_outputs_equal": fresh_data_equal,
        "fresh_model_outputs_equal": bool(fresh_model_equal),
        "fresh_pipeline_outputs_equal": fresh_pipeline_equal,
        "raw_records_present": len(records) > 0,
        "plot_present": (run_dir / "parallel_comparison.png").stat().st_size > 0,
        "status_passed": load(run_dir / "status.json")["passed"] is True,
    }
    result = {"passed": all(checks.values()), "check_count": len(checks), "checks": checks}
    (run_dir / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
