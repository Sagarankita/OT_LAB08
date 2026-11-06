"""Independently verify a completed Track B result and Track C QUBO handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.benchmarking import load_completed_run, sha256_file
from optimization_lab.discrete import (
    Item, enumerate_binary_model, exhaustive_knapsack, sufficient_cardinality_penalty,
)

METHODS = {"greedy", "dynamic_programming", "ilp", "branch_and_bound"}
EXACT = METHODS - {"greedy"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    loaded = load_completed_run(run)
    config, metadata, records, summary = (
        loaded["config"], loaded["metadata"], loaded["records"], loaded["summary"]
    )
    instances = json.loads((run / "knapsack_instances.json").read_text())
    oracle_checks = []
    per_seed_optimum = {}
    for instance in instances:
        items = [Item(item["value"], item["weight"]) for item in instance["items"]]
        oracle = exhaustive_knapsack(items, instance["capacity"])
        saved = instance["oracle"]
        oracle_checks.append(
            oracle.total_value == saved["total_value"] and
            oracle.total_weight == saved["total_weight"] and
            list(oracle.indices) == saved["indices"]
        )
        per_seed_optimum[instance["seed"]] = oracle.total_value
    expected_records = len(config["seeds"]) * config["repetitions"] * len(METHODS)
    paired = all(
        {record["method"] for record in records
         if record["seed"] == seed and record["repetition"] == repetition} == METHODS
        for seed in config["seeds"] for repetition in range(config["repetitions"])
    )
    exact_correct = all(
        record["objective_value"] == per_seed_optimum[record["seed"]]
        for record in records if record["method"] in EXACT
    )
    gaps_correct = all(math.isclose(
        record["optimality_gap_percent"],
        100.0 * (record["oracle_value"] - record["objective_value"]) / record["oracle_value"],
        abs_tol=1e-12,
    ) for record in records)

    source = json.loads((ROOT / "data/track_b_selection_source.json").read_text())
    frozen_bytes = (run / "track_b_qubo.json").read_bytes()
    frozen = json.loads(frozen_bytes)
    project_frozen = ROOT / "data/track_b_qubo.json"
    sufficient_penalty = sufficient_cardinality_penalty(source["benefits"], source["synergies"])
    sufficient = enumerate_binary_model(
        source["benefits"], source["synergies"], source["cardinality"], sufficient_penalty
    )
    insufficient = enumerate_binary_model(
        source["benefits"], source["synergies"], source["cardinality"],
        float(source["insufficient_penalty"]),
    )
    energy_equalities = all(
        math.isclose(state["expected_energy"], state["qubo_energy"], abs_tol=1e-9)
        and math.isclose(state["qubo_energy"], state["ising_energy"], abs_tol=1e-9)
        for state in sufficient["states"] + insufficient["states"]
    )
    status = json.loads((run / "status.json").read_text())
    artifact_hashes = all(
        sha256_file(run / artifact["path"]) == artifact["sha256"]
        for artifact in status["artifacts"].values()
    )
    checks = {
        "final_configuration": config["run_kind"] == "final",
        "five_seed_instances": len(config["seeds"]) == len(instances) >= 5,
        "record_count": len(records) == expected_records == summary["record_count"],
        "all_method_pairs_present": paired,
        "all_selections_feasible": all(record["feasible"] and
                                        record["total_weight"] <= record["capacity"]
                                        for record in records),
        "independent_oracles_match": all(oracle_checks),
        "exact_methods_match_oracle": exact_correct,
        "greedy_never_exceeds_oracle": all(
            record["objective_value"] <= record["oracle_value"]
            for record in records if record["method"] == "greedy"
        ),
        "reported_gaps_recomputed": gaps_correct,
        "timing_policy_recorded": metadata["timing_policy"]["setup_included_in_execution"] is False,
        "frozen_qubo_identical_to_project_handoff": project_frozen.read_bytes() == frozen_bytes,
        "frozen_qubo_hash_matches_metadata": hashlib.sha256(frozen_bytes).hexdigest()
                                             == metadata["qubo_sha256"],
        "sufficient_penalty_recomputed": frozen["penalty_proof"]["chosen_penalty"]
                                         == sufficient_penalty,
        "qubo_and_ising_equal_for_all_states": energy_equalities,
        "sufficient_global_minimizers_feasible": all(
            state["feasible"] for state in sufficient["minimizers"]
        ),
        "insufficient_penalty_failure_demonstrated": any(
            not state["feasible"] for state in insufficient["minimizers"]
        ),
        "classical_optimum_matches_frozen": math.isclose(
            sufficient["minimum_energy"], frozen["classical_optimum"]["minimum_energy"],
            abs_tol=1e-9,
        ) and sufficient["minimizers"] == frozen["classical_optimum"]["minimizers"],
        "all_declared_artifact_hashes_match": artifact_hashes,
        "git_provenance_pending_user": metadata["provenance"]["git_revision"]
                                       == "pending user Git work",
    }
    report = {
        "checks": checks, "passed": all(checks.values()),
        "knapsack_instances": len(instances), "raw_records": len(records),
        "qubo_states_checked_per_penalty": len(sufficient["states"]),
    }
    (run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
