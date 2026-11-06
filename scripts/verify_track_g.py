"""Independently audit Track G source, patch, workloads and xv6 guest evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "vendor/xv6-riscv-baseline"
BASE = ROOT / "vendor/xv6-riscv-track-g-baseline"
STRIDE = ROOT / "vendor/xv6-riscv-track-g-stride"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_tests() -> list[str]:
    text = re.sub(r"//[^\n]*", "", (PINNED / "user/usertests.c").read_text())
    expected = []
    for array in ("quicktests", "slowtests"):
        match = re.search(r"\b" + array + r"\[\]\s*=\s*\{(.*?)\n\};", text, re.S)
        if not match:
            raise RuntimeError(f"cannot find {array}")
        expected.extend(re.findall(r'\{\s*\w+\s*,\s*"([^"]+)"\s*\}', match.group(1)))
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    metadata = json.loads((run / "metadata.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    records = json.loads((run / "records.json").read_text())
    analysis = json.loads((run / "analysis.json").read_text())
    baseline_console = (run / "console_baseline.log").read_text(errors="replace").replace("\r", "")
    stride_console = (run / "console_stride.log").read_text(errors="replace").replace("\r", "")
    expected = expected_tests()
    observed = re.findall(r"test ([a-zA-Z0-9_]+):", stride_console)
    policies = {"baseline", "stride"}
    expected_keys = {(workload, role, repetition) for repetition in range(3)
                     for workload, roles in (("cpu", 3), ("mixed", 2), ("latency", 3))
                     for role in range(roles)}
    per_policy_keys = {
        policy: {(r["workload"], r["role"], r["repetition"])
                 for r in records if r["policy"] == policy} for policy in policies
    }
    paired_checksums = all(
        len({r["checksum"] for r in records if
             (r["workload"], r["role"], r["repetition"]) == key}) == 1
        for key in expected_keys
    )
    patch = run / "xv6-stride.patch"
    dry_run = subprocess.run(["patch", "--dry-run", "-p1", "-d", str(PINNED)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             input=patch.read_bytes())
    manifest = json.loads((ROOT / "vendor/xv6-source-manifest.json").read_text())
    pinned_unchanged = all((PINNED / name).is_file() and sha256(PINNED / name) == digest
                           for name, digest in manifest.items())
    hashes_match = all(sha256(run / name) == item["sha256"]
                       for name, item in metadata["artifacts"].items())
    checks = {
        "final_one_cpu_configuration": metadata["run_kind"] == "final" and
                                       metadata["emulated_cpus"] == 1,
        "three_repetitions": metadata["repetitions"] == 3,
        "both_builds_succeeded": all(item["returncode"] == 0
                                     for item in summary["builds"].values()),
        "record_count": len(records) == summary["record_count"] == 48,
        "all_workloads_paired": all(per_policy_keys[p] == expected_keys for p in policies),
        "outputs_preserved_by_checksum": paired_checksums and summary["checksums_identical"],
        "baseline_has_no_extension_stats": all(
            r["ticks"] == -1 and r["dispatches"] == -1
            for r in records if r["policy"] == "baseline"),
        "stride_stats_present": all(
            r["ticks"] >= 0 and r["dispatches"] > 0
            for r in records if r["policy"] == "stride"),
        "weight_controls_reflected": all(
            r["weight"] in {1, 3, 6, 8, 10} for r in records),
        "baseline_guest_completed": summary["guest"]["baseline"]["passed"],
        "modified_guest_completed": summary["guest"]["stride"]["passed"],
        "all_upstream_tests_started_in_order": observed == expected and len(expected) == 74,
        "all_tests_passed_and_shell_recovered":
            bool(re.search(r"^ALL TESTS PASSED$", stride_console, re.M)) and
            bool(re.search(r"^(?:\$ )?TRACK_G_STRIDE_DONE$", stride_console, re.M)),
        "no_guest_panic_or_failure": not re.search(
            r"panic:|SOME TESTS FAILED|BENCH_ERROR", baseline_console + stride_console),
        "benchmark_source_identical":
            (BASE / "user/schedbench.c").read_bytes() ==
            (STRIDE / "user/schedbench.c").read_bytes(),
        "pinned_source_manifest_unchanged": pinned_unchanged,
        "patch_hash_and_dry_run": sha256(patch) == summary["patch"]["sha256"] and
                                  dry_run.returncode == 0,
        "kernel_binaries_differ": summary["builds"]["baseline"]["kernel_sha256"] !=
                                  summary["builds"]["stride"]["kernel_sha256"],
        "timer_resolution_disclosed": "~0.1 seconds" in metadata["timer_resolution"],
        "analysis_reports_mixed_result":
            analysis["metrics"]["cpu_weight_6_elapsed"]["change_percent"] < 0 and
            analysis["metrics"]["latency_target_response"]["change_percent"] > 0,
        "all_artifact_hashes_match": hashes_match,
        "git_provenance_pending_user":
            metadata["provenance"]["git_revision"] == "pending user Git work",
    }
    report = {"checks": checks, "passed": all(checks.values()),
              "expected_tests": len(expected), "observed_tests": len(observed),
              "raw_records": len(records), "patch_dry_run_returncode": dry_run.returncode}
    (run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
