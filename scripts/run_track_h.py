"""Run Track H SQLite storage-budgeted index-selection experiment."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_lab.benchmarking import (
    BenchmarkConfig, RunRecorder, canonical_json_bytes, record_invalid_attempt,
)
from optimization_lab.database_optimization import (
    CANDIDATES, QUERIES, connect, create_dataset, create_indexes,
    dynamic_programming_select, execute_query, greedy_select, index_storage,
    query_plan, rows_hash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def configure_database(source: Path, destination: Path, indexes: list[str]):
    shutil.copy2(source, destination)
    connection = connect(destination)
    started = time.perf_counter()
    details = create_indexes(connection, indexes)
    build_seconds = time.perf_counter() - started
    storage = index_storage(connection, indexes)
    return connection, details, build_seconds, storage


def run_workload(connection, warmups: int, repetitions: int):
    for _ in range(warmups):
        for query in QUERIES:
            execute_query(connection, query)
    records = []
    workload_runs = []
    expected = {}
    for repetition in range(repetitions):
        total = 0.0
        for query in QUERIES:
            started = time.perf_counter()
            rows = execute_query(connection, query)
            elapsed = time.perf_counter() - started
            digest = rows_hash(rows)
            expected.setdefault(query.name, digest)
            if expected[query.name] != digest:
                raise AssertionError(f"non-deterministic output for {query.name}")
            records.append({"repetition": repetition, "query": query.name,
                            "execution_seconds": elapsed, "row_count": len(rows),
                            "result_sha256": digest})
            total += elapsed
        workload_runs.append({"repetition": repetition, "execution_seconds": total,
                              "throughput_queries_per_second": len(QUERIES) / total})
    return records, workload_runs, expected


def aggregate(values):
    return {"count": len(values), "mean": fmean(values),
            "sample_std": stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values), "maximum": max(values)}


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
               "--results-root", str(args.results_root.resolve())]
    try:
        raw_bytes = config_path.read_bytes()
        config = BenchmarkConfig.from_mapping(json.loads(raw_bytes))
        settings = config.solver_settings
        if len(config.seeds) != 1:
            raise ValueError("Track H uses exactly one deterministic dataset seed")
        if settings["variants"] != ["baseline", "greedy", "dynamic_programming", "all_indexes"]:
            raise ValueError("unexpected variant order")
        budget_fraction = float(settings["budget_fraction_of_individual_pages"])
        estimation_repetitions = int(settings["selection_estimation_repetitions"])
        if not 0 < budget_fraction < 1 or estimation_repetitions <= 0:
            raise ValueError("invalid selection settings")
    except Exception as exc:
        raw_bytes = config_path.read_bytes() if config_path.is_file() else b""
        directory = record_invalid_attempt(args.results_root.resolve(), config_path,
                                           raw_bytes, exc, command)
        print(f"INVALID: {exc}\nEvidence: {directory}", file=sys.stderr)
        return 2

    recorder = RunRecorder(
        args.results_root.resolve(), config_path, config,
        source_files=[Path(__file__).resolve(),
                      ROOT / "src/optimization_lab/database_optimization.py",
                      ROOT / "src/optimization_lab/benchmarking.py"], command=command,
    )
    total_started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="track_h_") as temporary:
            temp = Path(temporary)
            base = temp / "base.db"
            generation_started = time.perf_counter()
            counts = create_dataset(
                base, config.seeds[0], config.input_size,
                int(settings["customer_count"]), int(settings["product_count"]),
            )
            generation_seconds = time.perf_counter() - generation_started
            base_size = base.stat().st_size

            baseline_connection, _, _, _ = configure_database(base, temp / "estimate_base.db", [])
            baseline_raw, _, baseline_hashes = run_workload(
                baseline_connection, config.warmup_repetitions, estimation_repetitions
            )
            baseline_connection.close()
            baseline_total_by_rep = defaultdict(float)
            for record in baseline_raw:
                baseline_total_by_rep[record["repetition"]] += record["execution_seconds"]
            baseline_mean = fmean(baseline_total_by_rep.values())

            candidate_measurements = []
            for candidate in CANDIDATES:
                path = temp / f"estimate_{candidate.name}.db"
                connection, details, build_seconds, storage = configure_database(
                    base, path, [candidate.name]
                )
                raw, _, hashes = run_workload(
                    connection, config.warmup_repetitions, estimation_repetitions
                )
                plans = {query.name: query_plan(connection, query) for query in QUERIES}
                connection.close()
                if hashes != baseline_hashes:
                    raise AssertionError(f"candidate {candidate.name} changed query results")
                totals = defaultdict(float)
                for record in raw:
                    totals[record["repetition"]] += record["execution_seconds"]
                mean = fmean(totals.values())
                bytes_used = storage[candidate.name]
                page_size = sqlite3.connect(path).execute("PRAGMA page_size").fetchone()[0]
                pages = math.ceil(bytes_used / page_size)
                benefit_us = max(0, round((baseline_mean - mean) * 1_000_000))
                candidate_measurements.append({
                    "name": candidate.name, "sql": candidate.sql,
                    "storage_bytes": bytes_used, "pages": pages,
                    "build_seconds": build_seconds,
                    "baseline_mean_workload_seconds": baseline_mean,
                    "individual_mean_workload_seconds": mean,
                    "benefit_us": benefit_us, "raw_query_measurements": raw,
                    "plans": plans,
                })

            total_pages = sum(item["pages"] for item in candidate_measurements)
            budget_pages = max(1, math.floor(total_pages * budget_fraction))
            greedy = greedy_select(candidate_measurements, budget_pages)
            dp = dynamic_programming_select(candidate_measurements, budget_pages)
            all_names = [candidate.name for candidate in CANDIDATES]
            variants = {
                "baseline": [], "greedy": greedy,
                "dynamic_programming": dp, "all_indexes": all_names,
            }

            plans_artifact = {}
            variants_artifact = {}
            workload_runs_artifact = []
            expected_results = None
            measurement_cache = {}
            for variant, indexes in variants.items():
                physical_design = tuple(indexes)
                reused_from = None
                if physical_design in measurement_cache:
                    cached = measurement_cache[physical_design]
                    details, build_seconds, storage = cached["details"], cached["build_seconds"], cached["storage"]
                    plans_artifact[variant] = cached["plans"]
                    raw = [dict(record) for record in cached["raw"]]
                    workload_runs = [dict(record) for record in cached["workload_runs"]]
                    hashes = dict(cached["hashes"])
                    reused_from = cached["variant"]
                else:
                    connection, details, build_seconds, storage = configure_database(
                        base, temp / f"variant_{variant}.db", indexes
                    )
                    plans_artifact[variant] = {
                        query.name: query_plan(connection, query) for query in QUERIES
                    }
                    raw, workload_runs, hashes = run_workload(
                        connection, config.warmup_repetitions, config.repetitions
                    )
                    connection.close()
                    measurement_cache[physical_design] = {
                        "variant": variant, "details": details,
                        "build_seconds": build_seconds, "storage": storage,
                        "plans": plans_artifact[variant], "raw": raw,
                        "workload_runs": workload_runs, "hashes": hashes,
                    }
                if expected_results is None:
                    expected_results = hashes
                if hashes != expected_results:
                    raise AssertionError(f"variant {variant} changed query results")
                for record in raw:
                    recorder.add_record({"variant": variant, "selected_indexes": ";".join(indexes),
                                         **record})
                for record in workload_runs:
                    workload_runs_artifact.append({"variant": variant, **record})
                variants_artifact[variant] = {
                    "selected_indexes": indexes, "selection_pages": sum(
                        item["pages"] for item in candidate_measurements
                        if item["name"] in indexes),
                    "budget_pages": budget_pages if variant not in {"baseline", "all_indexes"}
                                      else None,
                    "measured_index_storage_bytes": sum(storage.values()),
                    "per_index_storage_bytes": storage,
                    "index_build_seconds": build_seconds,
                    "index_definitions": details,
                    "measurement_reused_from": reused_from,
                }
            if time.perf_counter() - total_started > config.timeout_seconds:
                raise TimeoutError(f"Track H exceeded {config.timeout_seconds} seconds")

            candidate_path = recorder.directory / "candidate_measurements.json"
            plans_path = recorder.directory / "query_plans.json"
            variants_path = recorder.directory / "variants.json"
            workload_path = recorder.directory / "workload_runs.json"
            results_path = recorder.directory / "expected_results.json"
            dataset_path = recorder.directory / "dataset_manifest.json"
            write_json(candidate_path, {
                "baseline_raw_query_measurements": baseline_raw,
                "baseline_mean_workload_seconds": baseline_mean,
                "budget_fraction": budget_fraction, "budget_pages": budget_pages,
                "total_individual_pages": total_pages, "candidates": candidate_measurements,
                "greedy_selection": greedy, "dynamic_programming_selection": dp,
            })
            write_json(plans_path, plans_artifact)
            write_json(variants_path, variants_artifact)
            write_json(workload_path, workload_runs_artifact)
            write_json(results_path, expected_results)
            write_json(dataset_path, {
                "seed": config.seeds[0], "counts": counts,
                "generation_formula": "deterministic modular arithmetic in create_dataset",
                "base_database_bytes": base_size,
                "generation_seconds": generation_seconds,
                "sqlite_version": sqlite3.sqlite_version,
                "page_size": sqlite3.connect(base).execute("PRAGMA page_size").fetchone()[0],
                "query_count": len(QUERIES),
            })

            per_variant = {}
            for variant in variants:
                query_records = [r for r in recorder.records if r["variant"] == variant]
                workload_values = [r["execution_seconds"] for r in workload_runs_artifact
                                   if r["variant"] == variant]
                throughput_values = [r["throughput_queries_per_second"]
                                     for r in workload_runs_artifact if r["variant"] == variant]
                per_variant[variant] = {
                    "workload_seconds": aggregate(workload_values),
                    "throughput_queries_per_second": aggregate(throughput_values),
                    "query_seconds": {
                        query.name: aggregate([r["execution_seconds"] for r in query_records
                                               if r["query"] == query.name])
                        for query in QUERIES
                    },
                    **variants_artifact[variant],
                }

            plot = recorder.directory / "database_comparison.png"
            labels = list(variants)
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            x = list(range(len(QUERIES)))
            width = 0.19
            for offset, variant in enumerate(labels):
                values = [per_variant[variant]["query_seconds"][q.name]["mean"] * 1000
                          for q in QUERIES]
                axes[0].bar([i + (offset - 1.5) * width for i in x], values, width,
                            label=variant)
            axes[0].set_xticks(x, [q.name for q in QUERIES], rotation=28, ha="right")
            axes[0].set_yscale("log")
            axes[0].set_ylabel("Mean latency (ms, log scale)")
            axes[0].set_title("Per-query latency")
            axes[0].legend(fontsize=7)
            axes[1].bar(labels, [per_variant[v]["throughput_queries_per_second"]["mean"]
                                 for v in labels])
            axes[1].set_ylabel("Queries per second")
            axes[1].set_title("Complete workload throughput")
            axes[1].tick_params(axis="x", rotation=20)
            axes[2].bar(labels, [per_variant[v]["measured_index_storage_bytes"] / 1048576
                                 for v in labels])
            axes[2].axhline(budget_pages * 4096 / 1048576, color="red", linestyle="--",
                            label="selection budget")
            axes[2].set_ylabel("Added index storage (MiB)")
            axes[2].set_title("Storage cost")
            axes[2].tick_params(axis="x", rotation=20)
            axes[2].legend(fontsize=8)
            for axis in axes:
                axis.grid(axis="y", alpha=0.25)
            fig.suptitle("Track H — SQLite index selection on identical 100,000-order workload")
            fig.tight_layout()
            fig.savefig(plot, dpi=140)
            plt.close(fig)

            recorder.metadata["timing_policy"] = {
                "clock": "time.perf_counter", "warm_cache": True,
                "warmup_repetitions": config.warmup_repetitions,
                "measured_repetitions": config.repetitions,
                "connection_policy": "one connection per variant; same query order",
                "index_build_excluded_from_query_latency": True,
                "selection_estimation_repetitions": estimation_repetitions,
            }
            recorder.metadata["sqlite_pragmas"] = {
                "cache_size": -65536, "temp_store": "MEMORY", "automatic_index": "ON"
            }
            write_json(recorder.directory / "metadata.json", recorder.metadata)
            baseline_seconds = per_variant["baseline"]["workload_seconds"]["mean"]
            summary = {
                "status": "completed", "scope": "Track H SQLite index selection",
                "dataset_counts": counts, "query_count": len(QUERIES),
                "candidate_count": len(CANDIDATES), "record_count": len(recorder.records),
                "budget_pages": budget_pages, "budget_bytes": budget_pages * 4096,
                "selection_model": "0/1 knapsack over measured individual benefit and pages",
                "variants": per_variant,
                "workload_latency_improvement_percent": {
                    variant: 100.0 * (baseline_seconds - values["workload_seconds"]["mean"]) /
                             baseline_seconds for variant, values in per_variant.items()
                },
                "all_result_hashes_identical": True,
            }
            recorder.complete(summary, {
                "comparison_plot": plot, "candidate_measurements": candidate_path,
                "query_plans": plans_path, "variants": variants_path,
                "workload_runs": workload_path, "expected_results": results_path,
                "dataset_manifest": dataset_path,
            })
            print(recorder.directory)
            return 0
    except Exception as exc:
        recorder.fail(exc)
        print(f"FAILED: {exc}\nEvidence: {recorder.directory}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
