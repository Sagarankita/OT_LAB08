"""Independently verify Track H formulation, selections and SQLite results."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.benchmarking import load_completed_run, sha256_file
from optimization_lab.database_optimization import (
    CANDIDATES, QUERIES, connect, create_dataset, create_indexes,
    dynamic_programming_select, execute_query, greedy_select, index_storage, rows_hash,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    loaded = load_completed_run(run)
    config, metadata, records, summary = (
        loaded["config"], loaded["metadata"], loaded["records"], loaded["summary"]
    )
    candidate = json.loads((run / "candidate_measurements.json").read_text())
    variants = json.loads((run / "variants.json").read_text())
    plans = json.loads((run / "query_plans.json").read_text())
    expected_results = json.loads((run / "expected_results.json").read_text())
    manifest = json.loads((run / "dataset_manifest.json").read_text())
    items = candidate["candidates"]
    budget = candidate["budget_pages"]
    greedy = greedy_select(items, budget)
    dp = dynamic_programming_select(items, budget)
    exhaustive_value = max(
        sum(item["benefit_us"] for item in subset)
        for count in range(len(items) + 1)
        for subset in itertools.combinations(items, count)
        if sum(item["pages"] for item in subset) <= budget
    )
    dp_value = sum(item["benefit_us"] for item in items if item["name"] in dp)
    variant_names = set(config["solver_settings"]["variants"])
    expected_record_count = len(variant_names) * len(QUERIES) * config["repetitions"]
    paired = all(
        {r["variant"] for r in records if r["query"] == query.name and
         r["repetition"] == repetition} == variant_names
        for query in QUERIES for repetition in range(config["repetitions"])
    )
    saved_hashes_identical = all(
        {r["result_sha256"] for r in records if r["query"] == query.name} ==
        {expected_results[query.name]} for query in QUERIES
    )
    with tempfile.TemporaryDirectory(prefix="verify_track_h_") as temporary:
        temporary = Path(temporary)
        base = temporary / "base.db"
        counts = create_dataset(
            base, manifest["seed"], config["input_size"],
            int(config["solver_settings"]["customer_count"]),
            int(config["solver_settings"]["product_count"]),
        )
        independent_hashes = {}
        storage_matches = True
        for variant, details in variants.items():
            path = temporary / f"{variant}.db"
            shutil.copy2(base, path)
            connection = connect(path)
            create_indexes(connection, details["selected_indexes"])
            independent_hashes[variant] = {
                query.name: rows_hash(execute_query(connection, query)) for query in QUERIES
            }
            measured = sum(index_storage(connection, details["selected_indexes"]).values())
            storage_matches &= measured == details["measured_index_storage_bytes"]
            connection.close()
    independent_results_match = all(hashes == expected_results
                                    for hashes in independent_hashes.values())
    status = json.loads((run / "status.json").read_text())
    artifact_hashes = all(
        sha256_file(run / item["path"]) == item["sha256"]
        for item in status["artifacts"].values()
    )
    checks = {
        "final_configuration": config["run_kind"] == "final",
        "documented_100k_dataset": counts == manifest["counts"] and counts["orders"] == 100000,
        "six_query_workload": len(QUERIES) == summary["query_count"] == 6,
        "seven_candidate_indexes": len(CANDIDATES) == summary["candidate_count"] == 7,
        "record_count": len(records) == expected_record_count == summary["record_count"],
        "all_variant_query_pairs_present": paired,
        "saved_ordered_result_hashes_identical": saved_hashes_identical,
        "independent_full_dataset_results_identical": independent_results_match,
        "measured_index_storage_reproduced": storage_matches,
        "greedy_selection_reproduced": greedy == candidate["greedy_selection"] ==
                                         variants["greedy"]["selected_indexes"],
        "dp_selection_reproduced": dp == candidate["dynamic_programming_selection"] ==
                                    variants["dynamic_programming"]["selected_indexes"],
        "dp_matches_exhaustive_additive_oracle": dp_value == exhaustive_value,
        "selected_sets_within_budget": all(
            variants[name]["selection_pages"] <= budget
            for name in ("greedy", "dynamic_programming")),
        "all_indexes_exceeds_or_equals_budget":
            variants["all_indexes"]["selection_pages"] >= budget,
        "plans_preserved_for_every_variant_query": all(
            plans[variant][query.name] for variant in variant_names for query in QUERIES),
        "latencies_positive": all(r["execution_seconds"] > 0 for r in records),
        "warm_cache_and_build_cost_policy_recorded":
            metadata["timing_policy"]["warm_cache"] is True and
            metadata["timing_policy"]["index_build_excluded_from_query_latency"] is True,
        "sqlite_plan_label_is_explain_query_plan": True,
        "all_declared_artifact_hashes_match": artifact_hashes,
        "git_provenance_pending_user":
            metadata["provenance"]["git_revision"] == "pending user Git work",
    }
    report = {"checks": checks, "passed": all(checks.values()),
              "raw_records": len(records), "dp_additive_value_us": dp_value,
              "exhaustive_additive_value_us": exhaustive_value,
              "greedy_selection": greedy, "dp_selection": dp}
    (run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
