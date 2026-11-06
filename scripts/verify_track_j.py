#!/usr/bin/env python3
"""Independently verify saved Track J artifacts and selections."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.test_selection import (  # noqa: E402
    count_meaningful_python_lines,
    select_greedy,
    select_ilp,
    select_naive,
)


def load(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def line(value: str) -> tuple[str, int]:
    filename, number = value.rsplit(":", 1)
    return filename, int(number)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run_dir = args.run_directory.resolve()

    config = load(run_dir / "config.json")
    provenance = load(run_dir / "provenance.json")
    profiles = {
        method: load(run_dir / f"profile_{method}.json")
        for method in ("full", "naive", "greedy", "ilp")
    }
    selections = load(run_dir / "selections.json")
    benchmarks = load(run_dir / "benchmarks.json")
    mutations = load(run_dir / "mutations.json")
    summary = load(run_dir / "summary.json")

    codebase = ROOT / "vendor" / "track_j" / "more_itertools-10.7.0"
    archive = ROOT / "vendor" / "track_j" / "more_itertools-10.7.0.tar.gz"
    source_files = sorted((codebase / "more_itertools").glob("*.py"))
    meaningful = sum(count_meaningful_python_lines(path.read_text(encoding="utf-8")) for path in source_files)
    physical = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in source_files)

    full_target = {line(value) for value in profiles["full"]["executed_lines"]}
    coverage_map = {
        test: {line(value) for value in values}
        for test, values in profiles["full"]["per_test_coverage"].items()
    }
    durations = {test: float(value) for test, value in profiles["full"]["durations_seconds"].items()}
    selectable = set().union(*coverage_map.values())
    recomputed = {
        "naive": select_naive(coverage_map, durations, selectable),
        "greedy": select_greedy(coverage_map, durations, selectable),
        "ilp": select_ilp(
            coverage_map,
            durations,
            selectable,
            time_limit_seconds=float(config["ilp_time_limit_seconds"]),
        ),
    }

    checks = {
        "final_configuration": config["run_kind"] == "final",
        "pinned_version": provenance["project"] == "more-itertools" and provenance["version"] == "10.7.0",
        "archive_hash_reproduced": digest(archive) == provenance["archive_sha256"],
        "mit_license_present": provenance["license"] == "MIT" and (codebase / "LICENSE").exists(),
        "meaningful_source_over_1000": meaningful >= 1000 and meaningful == provenance["meaningful_nonblank_noncomment_source_lines"],
        "physical_source_count_reproduced": physical == provenance["physical_source_lines"],
        "full_suite_671_tests": len(durations) == 671 and summary["full_tests"] == 671,
        "full_target_2030_lines": len(full_target) == 2030 and summary["full_covered_lines"] == 2030,
        "all_profiled_suites_pass": all(profile["exit_code"] == 0 for profile in profiles.values()),
        "all_profiles_preserve_target": all(
            full_target <= {line(value) for value in profile["executed_lines"]}
            for profile in profiles.values()
        ),
        "naive_reproduced": list(recomputed["naive"].selected) == selections["naive"]["selected"],
        "greedy_reproduced": list(recomputed["greedy"].selected) == selections["greedy"]["selected"],
        "ilp_reproduced": list(recomputed["ilp"].selected) == selections["ilp"]["selected"],
        "ilp_optimal": recomputed["ilp"].solver_status == "optimal" and selections["ilp"]["solver_status"] == "optimal",
        "ilp_objective_not_worse": recomputed["ilp"].objective_seconds <= recomputed["greedy"].objective_seconds + 1e-9,
        "three_measured_runs": all(len(benchmarks[method]["samples_seconds"]) == 3 for method in benchmarks),
        "positive_benchmark_samples": all(
            sample > 0 for method in benchmarks.values() for sample in method["samples_seconds"]
        ),
        "reduced_suites_faster": all(
            summary["methods"][method]["mean_runtime_seconds"] < summary["methods"]["full"]["mean_runtime_seconds"]
            for method in ("naive", "greedy", "ilp")
        ),
        "two_mutations_four_methods": len(mutations) == 8,
        "all_seeded_mutations_killed": all(record["killed"] for record in mutations),
        "plot_present": (run_dir / "test_selection_comparison.png").stat().st_size > 0,
        "status_passed": load(run_dir / "status.json")["passed"] is True,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "check_count": len(checks),
        "full_tests": len(durations),
        "covered_lines": len(full_target),
        "selected_test_counts": {method: selections[method]["test_count"] for method in selections},
    }
    (run_dir / "verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
