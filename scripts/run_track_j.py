#!/usr/bin/env python3
"""Run Track J: coverage-preserving test-suite minimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from optimization_lab.test_selection import (  # noqa: E402
    Line,
    SelectionResult,
    count_meaningful_python_lines,
    select_greedy,
    select_ilp,
    select_naive,
)


CODEBASE = ROOT / "vendor" / "track_j" / "more_itertools-10.7.0"
ARCHIVE = ROOT / "vendor" / "track_j" / "more_itertools-10.7.0.tar.gz"
PROFILE_SCRIPT = ROOT / "scripts" / "profile_track_j_suite.py"
MUTATIONS = [
    {
        "id": "take_off_by_one",
        "file": "more_itertools/recipes.py",
        "old": "return list(islice(iterable, n))",
        "new": "return list(islice(iterable, max(0, n - 1)))",
    },
    {
        "id": "quantify_plus_one",
        "file": "more_itertools/recipes.py",
        "old": "return sum(map(pred, iterable))",
        "new": "return sum(map(pred, iterable)) + 1",
    },
]


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_line(value: str) -> Line:
    filename, line = value.rsplit(":", 1)
    return filename, int(line)


def run_profile(output: Path, selection: list[str] | None = None) -> dict:
    selection_file = None
    command = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--codebase",
        str(CODEBASE),
        "--output",
        str(output),
    ]
    if selection is not None:
        selection_file = output.with_name(output.stem + "_selection.json")
        dump_json(selection_file, selection)
        command.extend(["--selection", str(selection_file)])
    started = perf_counter()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    profile_wall_seconds = perf_counter() - started
    output.with_suffix(".stdout.txt").write_text(completed.stdout, encoding="utf-8")
    output.with_suffix(".stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError(f"coverage profile failed: {completed.stderr[-1000:]}")
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["profile_wall_seconds"] = profile_wall_seconds
    dump_json(output, payload)
    return payload


def benchmark_suite(
    selection: list[str] | None,
    repetitions: int,
    warmups: int,
    codebase: Path = CODEBASE,
) -> list[float]:
    target = selection if selection is not None else ["tests"]
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--disable-warnings",
        "--rootdir=.",
        *target,
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "20261028"
    samples: list[float] = []
    for iteration in range(warmups + repetitions):
        started = perf_counter()
        completed = subprocess.run(
            command,
            cwd=codebase,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        elapsed = perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(f"suite failed: {completed.stdout[-1000:]}\n{completed.stderr[-1000:]}")
        if iteration >= warmups:
            samples.append(elapsed)
    return samples


def run_mutations(selections: dict[str, list[str] | None], count: int) -> list[dict]:
    records: list[dict] = []
    for mutation in MUTATIONS[:count]:
        with tempfile.TemporaryDirectory(prefix="track-j-mutant-") as temp:
            mutant = Path(temp) / "more_itertools-10.7.0"
            shutil.copytree(CODEBASE, mutant)
            path = mutant / mutation["file"]
            original = path.read_text(encoding="utf-8")
            if original.count(mutation["old"]) != 1:
                raise RuntimeError(f"mutation site is not unique: {mutation['id']}")
            path.write_text(original.replace(mutation["old"], mutation["new"]), encoding="utf-8")
            for method, selection in selections.items():
                target = selection if selection is not None else ["tests"]
                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--disable-warnings",
                    "--rootdir=.",
                    *target,
                ]
                started = perf_counter()
                completed = subprocess.run(
                    command,
                    cwd=mutant,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=180,
                )
                records.append(
                    {
                        "mutation": mutation["id"],
                        "method": method,
                        "killed": completed.returncode != 0,
                        "exit_code": completed.returncode,
                        "runtime_seconds": perf_counter() - started,
                    }
                )
    return records


def selection_payload(result: SelectionResult) -> dict:
    return {
        "method": result.method,
        "selected": list(result.selected),
        "test_count": len(result.selected),
        "objective_seconds": result.objective_seconds,
        "selection_overhead_seconds": result.overhead_seconds,
        "covered_selectable_lines": result.covered_lines,
        "solver_status": result.solver_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not CODEBASE.exists() or not ARCHIVE.exists():
        raise FileNotFoundError("pinned more-itertools source and archive are required")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = ROOT / "results" / "track_j_test_selection" / timestamp
    run_dir.mkdir(parents=True)
    dump_json(run_dir / "config.json", config)

    source_files = sorted((CODEBASE / "more_itertools").glob("*.py"))
    physical_lines = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in source_files)
    meaningful_lines = sum(count_meaningful_python_lines(path.read_text(encoding="utf-8")) for path in source_files)
    provenance = {
        "project": "more-itertools",
        "version": "10.7.0",
        "upstream": "https://github.com/more-itertools/more-itertools",
        "source_distribution": "https://files.pythonhosted.org/packages/source/m/more-itertools/more_itertools-10.7.0.tar.gz",
        "archive_sha256": sha256(ARCHIVE),
        "license": "MIT",
        "license_file": "vendor/track_j/more_itertools-10.7.0/LICENSE",
        "production_python_files": [path.relative_to(CODEBASE).as_posix() for path in source_files],
        "physical_source_lines": physical_lines,
        "meaningful_nonblank_noncomment_source_lines": meaningful_lines,
    }
    dump_json(run_dir / "provenance.json", provenance)

    full_profile = run_profile(run_dir / "profile_full.json")
    if full_profile["exit_code"] != 0:
        raise RuntimeError("full profiling suite did not pass")
    full_target = {parse_line(line) for line in full_profile["executed_lines"]}
    durations = {test: float(value) for test, value in full_profile["durations_seconds"].items()}
    coverage_map = {
        test: {parse_line(line) for line in lines}
        for test, lines in full_profile["per_test_coverage"].items()
    }
    per_test_union = set().union(*coverage_map.values())
    common_lines = full_target - per_test_union
    selectable_target = full_target - common_lines

    naive = select_naive(coverage_map, durations, selectable_target)
    greedy = select_greedy(coverage_map, durations, selectable_target)
    ilp = select_ilp(
        coverage_map,
        durations,
        selectable_target,
        time_limit_seconds=float(config["ilp_time_limit_seconds"]),
    )
    selection_results = {result.method: result for result in (naive, greedy, ilp)}
    profiles = {"full": full_profile}
    selection_lists = {method: list(result.selected) for method, result in selection_results.items()}
    verification_repairs: dict[str, list[str]] = {method: [] for method in selection_lists}
    for method, selection in selection_lists.items():
        for _attempt in range(3):
            profile = run_profile(run_dir / f"profile_{method}.json", selection)
            retained = {parse_line(line) for line in profile["executed_lines"]}
            missing = full_target - retained
            if not missing:
                profiles[method] = profile
                break
            candidates = []
            for test in sorted(set(coverage_map) - set(selection)):
                gain = len(coverage_map[test] & missing)
                if gain:
                    candidates.append((gain / max(durations[test], 1e-9), gain, test))
            if not candidates:
                raise AssertionError(f"{method} has an unrecoverable coverage replay gap")
            repair = max(candidates)[2]
            selection.append(repair)
            selection.sort()
            verification_repairs[method].append(repair)
        else:
            raise AssertionError(f"{method} coverage replay remained unstable after repairs")

    dump_json(
        run_dir / "selections.json",
        {
            "full": {
                "test_count": len(durations),
                "selected": sorted(durations),
                "objective_seconds": sum(durations.values()),
                "selection_overhead_seconds": 0.0,
                "solver_status": "not_applicable",
                "verification_repairs": [],
            },
            **{
                method: {
                    **selection_payload(result),
                    "selected": selection_lists[method],
                    "test_count": len(selection_lists[method]),
                    "verification_repairs": verification_repairs[method],
                    "objective_seconds_after_repair": sum(
                        durations[test] for test in selection_lists[method]
                    ),
                }
                for method, result in selection_results.items()
            },
        },
    )

    profile_checks = {}
    for method, profile in profiles.items():
        retained = {parse_line(line) for line in profile["executed_lines"]}
        profile_checks[method] = {
            "exit_code": profile["exit_code"],
            "executed_lines": len(retained),
            "target_lines_retained": len(full_target & retained),
            "target_lines_total": len(full_target),
            "coverage_preserved": full_target <= retained,
        }
        if profile["exit_code"] != 0 or not full_target <= retained:
            raise AssertionError(f"{method} failed coverage-preserving rerun")

    repetitions = int(config["benchmark_repetitions"])
    warmups = int(config["benchmark_warmups"])
    benchmark_samples: dict[str, list[float]] = {}
    measurement_owner: dict[tuple[str, ...], str] = {}
    for method in ("full", "naive", "greedy", "ilp"):
        selection = None if method == "full" else selection_lists[method]
        key = tuple(selection or ["__full__"])
        if key in measurement_owner:
            benchmark_samples[method] = benchmark_samples[measurement_owner[key]]
        else:
            benchmark_samples[method] = benchmark_suite(selection, repetitions, warmups)
            measurement_owner[key] = method

    benchmark_summary = {}
    for method, samples in benchmark_samples.items():
        benchmark_summary[method] = {
            "samples_seconds": samples,
            "mean_seconds": statistics.mean(samples),
            "stdev_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "measurement_reused_from": measurement_owner.get(
                tuple((selection_lists.get(method) or ["__full__"])), method
            ),
        }
    dump_json(run_dir / "benchmarks.json", benchmark_summary)

    mutation_selections = {"full": None, **selection_lists}
    mutation_records = run_mutations(mutation_selections, int(config["mutations"]))
    dump_json(run_dir / "mutations.json", mutation_records)

    counts = [
        len(durations),
        len(selection_lists["naive"]),
        len(selection_lists["greedy"]),
        len(selection_lists["ilp"]),
    ]
    means = [benchmark_summary[name]["mean_seconds"] for name in ("full", "naive", "greedy", "ilp")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    labels = ["Full", "Naive", "Greedy", "ILP"]
    axes[0].bar(labels, counts, color=["#6b7280", "#60a5fa", "#34d399", "#f59e0b"])
    axes[0].set_ylabel("Selected tests")
    axes[0].set_title("Suite size")
    axes[1].bar(labels, means, color=["#6b7280", "#60a5fa", "#34d399", "#f59e0b"])
    axes[1].set_ylabel("Wall-clock seconds")
    axes[1].set_title(f"Execution mean ({repetitions} measured runs)")
    fig.suptitle("Track J: coverage-preserving test selection")
    fig.tight_layout()
    fig.savefig(run_dir / "test_selection_comparison.png", dpi=160)
    plt.close(fig)

    full_mean = benchmark_summary["full"]["mean_seconds"]
    summary = {
        "run_kind": config["run_kind"],
        "codebase": provenance,
        "full_tests": len(durations),
        "full_covered_lines": len(full_target),
        "session_common_lines": len(common_lines),
        "selectable_target_lines": len(selectable_target),
        "profile_checks": profile_checks,
        "profiling_wall_seconds": {
            method: profiles[method]["profile_wall_seconds"] for method in profiles
        },
        "total_profiling_wall_seconds": sum(
            profiles[method]["profile_wall_seconds"] for method in profiles
        ),
        "methods": {},
        "mutations": mutation_records,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor_count": os.cpu_count(),
        },
    }
    for method in ("full", "naive", "greedy", "ilp"):
        tests = len(durations) if method == "full" else len(selection_lists[method])
        mean = benchmark_summary[method]["mean_seconds"]
        summary["methods"][method] = {
            "tests": tests,
            "mean_runtime_seconds": mean,
            "stdev_runtime_seconds": benchmark_summary[method]["stdev_seconds"],
            "runtime_reduction_percent": 100.0 * (full_mean - mean) / full_mean,
            "selection_overhead_seconds": 0.0 if method == "full" else selection_results[method].overhead_seconds,
            "solver_status": "not_applicable" if method == "full" else selection_results[method].solver_status,
            "coverage_preserved": profile_checks[method]["coverage_preserved"],
            "verification_repairs": [] if method == "full" else verification_repairs[method],
        }
    dump_json(run_dir / "summary.json", summary)
    dump_json(run_dir / "status.json", {"passed": True, "run_directory": run_dir.relative_to(ROOT).as_posix()})
    latest = ROOT / "results" / "track_j_test_selection" / "latest-success.json"
    dump_json(
        latest,
        {
            "run_directory": run_dir.relative_to(ROOT).as_posix(),
            "run_kind": config["run_kind"],
            "passed": True,
            "full_tests": len(durations),
            "full_covered_lines": len(full_target),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
