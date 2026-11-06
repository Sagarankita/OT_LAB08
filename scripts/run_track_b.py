"""Run Track B knapsack comparisons and freeze the Track C QUBO input."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
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
from optimization_lab.discrete import (
    Item, branch_and_bound_knapsack, dynamic_programming_knapsack,
    enumerate_binary_model, exhaustive_knapsack, generate_knapsack,
    greedy_knapsack, ilp_knapsack, sufficient_cardinality_penalty,
)

METHODS = {
    "greedy": greedy_knapsack,
    "dynamic_programming": dynamic_programming_knapsack,
    "ilp": ilp_knapsack,
    "branch_and_bound": branch_and_bound_knapsack,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


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
        capacity_ratio = float(settings["capacity_ratio"])
        if config.input_size > 24:
            raise ValueError("Track B final instances must have at most 24 items for exhaustive validation")
    except Exception as exc:
        raw_bytes = config_path.read_bytes() if config_path.is_file() else b""
        directory = record_invalid_attempt(args.results_root.resolve(), config_path,
                                           raw_bytes, exc, command)
        print(f"INVALID: {exc}\nEvidence: {directory}", file=sys.stderr)
        return 2

    source_model_path = ROOT / "data/track_b_selection_source.json"
    recorder = RunRecorder(
        args.results_root.resolve(), config_path, config,
        source_files=[Path(__file__).resolve(), ROOT / "src/optimization_lab/discrete.py",
                      ROOT / "src/optimization_lab/benchmarking.py", source_model_path],
        command=command,
    )
    total_started = time.perf_counter()
    try:
        instance_records = []
        instances: dict[int, tuple[list[Item], int]] = {}
        setup_started = time.perf_counter()
        for seed in config.seeds:
            items, capacity = generate_knapsack(seed, config.input_size, capacity_ratio)
            oracle_started = time.perf_counter()
            oracle = exhaustive_knapsack(items, capacity)
            oracle_seconds = time.perf_counter() - oracle_started
            instances[seed] = (items, capacity)
            instance_records.append({
                "seed": seed,
                "items": [{"value": item.value, "weight": item.weight} for item in items],
                "capacity": capacity,
                "capacity_ratio_rule": capacity_ratio,
                "oracle": {
                    "method": "exhaustive_enumeration",
                    "indices": list(oracle.indices), "total_value": oracle.total_value,
                    "total_weight": oracle.total_weight,
                    "subsets": oracle.details["subsets"],
                    "feasible_subsets": oracle.details["feasible_subsets"],
                    "execution_seconds": oracle_seconds,
                },
            })
        for seed in config.seeds:
            items, capacity = instances[seed]
            for _ in range(config.warmup_repetitions):
                for solve in METHODS.values():
                    solve(items, capacity)
        setup_seconds = time.perf_counter() - setup_started
        deadline = total_started + config.timeout_seconds
        for instance in instance_records:
            seed = instance["seed"]
            items, capacity = instances[seed]
            optimum = instance["oracle"]["total_value"]
            for repetition in range(config.repetitions):
                for name, solve in METHODS.items():
                    if time.perf_counter() > deadline:
                        raise TimeoutError(f"Track B exceeded {config.timeout_seconds} seconds")
                    started = time.perf_counter()
                    solution = solve(items, capacity)
                    elapsed = time.perf_counter() - started
                    if solution.total_weight > capacity:
                        raise AssertionError(f"{name} returned an infeasible selection")
                    if name != "greedy" and solution.total_value != optimum:
                        raise AssertionError(
                            f"exact method {name} value {solution.total_value} != oracle {optimum}"
                        )
                    gap = 0.0 if optimum == 0 else 100.0 * (optimum - solution.total_value) / optimum
                    recorder.add_record({
                        "seed": seed, "repetition": repetition, "method": name,
                        "item_count": len(items), "capacity": capacity,
                        "selected_indices": ";".join(str(i) for i in solution.indices),
                        "objective_value": solution.total_value,
                        "total_weight": solution.total_weight,
                        "feasible": True, "oracle_value": optimum,
                        "optimality_gap_percent": gap,
                        "execution_seconds": elapsed,
                        "solver_status": solution.details.get("status", "not_applicable"),
                        "search_nodes": solution.details.get(
                            "nodes_expanded", solution.details.get("mip_node_count", "not_available")
                        ),
                    })

        source = json.loads(source_model_path.read_text())
        benefits = source["benefits"]
        synergies = source["synergies"]
        cardinality = source["cardinality"]
        sufficient_penalty = sufficient_cardinality_penalty(benefits, synergies)
        insufficient_penalty = float(source["insufficient_penalty"])
        sufficient = enumerate_binary_model(benefits, synergies, cardinality, sufficient_penalty)
        insufficient = enumerate_binary_model(benefits, synergies, cardinality, insufficient_penalty)
        if not all(state["feasible"] for state in sufficient["minimizers"]):
            raise AssertionError("sufficient penalty produced an infeasible global minimizer")
        if not any(not state["feasible"] for state in insufficient["minimizers"]):
            raise AssertionError("insufficient penalty did not demonstrate constraint failure")
        reward_upper_bound = sufficient_penalty - 1.0
        frozen_qubo = {
            "schema_version": 1,
            "source_problem": source,
            "source_sha256": hashlib.sha256(source_model_path.read_bytes()).hexdigest(),
            "penalty_proof": {
                "nonnegative_reward_upper_bound": reward_upper_bound,
                "chosen_penalty": sufficient_penalty,
                "argument": (
                    "Any infeasible state violates cardinality by at least one, so its energy is "
                    "at least -reward_upper_bound+penalty > 0; every feasible state's energy is "
                    "-score <= 0. Therefore every global minimizer is feasible."
                ),
            },
            "qubo": sufficient["qubo"],
            "classical_optimum": {
                "minimum_energy": sufficient["minimum_energy"],
                "minimizers": sufficient["minimizers"],
            },
            "insufficient_penalty_demo": {
                "penalty": insufficient_penalty,
                "minimum_energy": insufficient["minimum_energy"],
                "minimizers": insufficient["minimizers"],
            },
            "track_c_instruction": (
                "Use these exact QUBO coefficients, offset, binary-to-spin convention and classical "
                "optimum for QAOA; do not regenerate or rescale without recording a new artifact."
            ),
        }
        frozen_bytes = canonical_json_bytes(frozen_qubo)
        frozen_path = ROOT / "data/track_b_qubo.json"
        frozen_path.write_bytes(frozen_bytes)
        run_qubo_path = recorder.directory / "track_b_qubo.json"
        run_qubo_path.write_bytes(frozen_bytes)
        enumeration_path = recorder.directory / "qubo_enumeration.json"
        write_json(enumeration_path, {
            "sufficient_penalty_states": sufficient["states"],
            "insufficient_penalty_states": insufficient["states"],
        })
        instances_path = recorder.directory / "knapsack_instances.json"
        write_json(instances_path, instance_records)

        timing = numeric_summary(recorder.records, "method", "execution_seconds")
        gaps = numeric_summary(recorder.records, "method", "optimality_gap_percent")
        methods = list(METHODS)
        figure = recorder.directory / "method_comparison.png"
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        axes[0].bar(methods, [timing[name]["mean"] for name in methods],
                    yerr=[timing[name]["sample_std"] for name in methods], capsize=4)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Execution time (seconds; log scale)")
        axes[0].set_title("Runtime: mean ± sample SD")
        axes[1].bar(methods, [gaps[name]["mean"] for name in methods])
        axes[1].set_ylabel("Optimality gap (%)")
        axes[1].set_title("Solution quality vs exhaustive oracle")
        for axis in axes:
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.25)
        fig.suptitle("Track B knapsack — identical seeded instances")
        fig.tight_layout()
        fig.savefig(figure, dpi=140)
        plt.close(fig)

        recorder.metadata["timing_policy"] = {
            "setup_included_in_execution": False,
            "setup_seconds": setup_seconds,
            "oracle_timing_in_setup": True,
            "warmup_repetitions_per_seed_and_method": config.warmup_repetitions,
            "measured_repetitions_per_seed_and_method": config.repetitions,
            "clock": "time.perf_counter",
        }
        recorder.metadata["qubo_sha256"] = hashlib.sha256(frozen_bytes).hexdigest()
        (recorder.directory / "metadata.json").write_bytes(canonical_json_bytes(recorder.metadata))
        exact_failures = sum(
            record["objective_value"] != record["oracle_value"]
            for record in recorder.records if record["method"] != "greedy"
        )
        summary = {
            "status": "completed", "scope": "Track B discrete optimization",
            "baseline": "value-to-weight greedy", "instance_count": len(config.seeds),
            "item_count": config.input_size, "record_count": len(recorder.records),
            "repetitions": config.repetitions, "setup_seconds": setup_seconds,
            "timing_by_method": timing, "optimality_gap_percent_by_method": gaps,
            "exact_method_oracle_mismatches": exact_failures,
            "qubo": {
                "variables": len(benefits), "cardinality": cardinality,
                "sufficient_penalty": sufficient_penalty,
                "sufficient_minimizers_all_feasible": True,
                "insufficient_penalty": insufficient_penalty,
                "insufficient_minimizer_infeasible": True,
                "minimum_energy": sufficient["minimum_energy"],
                "frozen_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
            },
        }
        recorder.complete(summary, {
            "comparison_plot": figure, "knapsack_instances": instances_path,
            "frozen_qubo": run_qubo_path, "qubo_enumeration": enumeration_path,
        })
        print(recorder.directory)
        return 0
    except Exception as exc:
        recorder.fail(exc)
        print(f"FAILED: {exc}\nEvidence: {recorder.directory}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

