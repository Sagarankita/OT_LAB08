"""Independently reload and verify a completed infrastructure smoke run."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    loaded = load_completed_run(run)
    config, metadata, records, summary = (
        loaded["config"], loaded["metadata"], loaded["records"], loaded["summary"]
    )
    expected_count = len(config["seeds"]) * config["repetitions"] * 2
    checks = {
        "final_run": config["run_kind"] == "final",
        "five_or_more_seeds": len(config["seeds"]) >= 5,
        "raw_record_count": len(records) == expected_count == summary["record_count"],
        "unique_seed_inputs": len({record["input_sha256"] for record in records})
                              == len(config["seeds"]),
        "all_records_valid": all(record["valid"] for record in records),
        "both_methods_each_repetition": all(
            {record["method"] for record in records
             if record["seed"] == seed and record["repetition"] == repetition}
            == {"python_fsum", "numpy_dot"}
            for seed in config["seeds"] for repetition in range(config["repetitions"])
        ),
        "objective_agreement": all(
            math.isclose(record["objective"], next(
                other["objective"] for other in records
                if other["seed"] == record["seed"]
                and other["repetition"] == record["repetition"]
                and other["method"] == "python_fsum"
            ), rel_tol=1e-10, abs_tol=1e-10)
            for record in records
        ),
        "setup_separate": metadata["timing_policy"]["setup_included_in_execution"] is False,
        "warmups_recorded": metadata["timing_policy"]["warmup_repetitions_per_seed_and_method"]
                            == config["warmup_repetitions"] >= 1,
        "units_recorded": metadata["units"] == config["units"],
        "solver_settings_recorded": metadata["solver_settings"] == config["solver_settings"],
        "config_hash_matches": hashlib.sha256(
            ((json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)) + "\n").encode()
        ).hexdigest() == metadata["config_sha256"],
        "plot_hash_matches": sha256_file(run / "timing_by_method.png")
                             == json.loads((run / "status.json").read_text())["artifacts"]["timing_plot"]["sha256"],
        "provenance_is_honest": metadata["provenance"]["git_revision"] == "pending user Git work",
        "scope_not_overclaimed": summary["scope"].startswith("infrastructure smoke only"),
    }
    report = {"checks": checks, "passed": all(checks.values()), "records": len(records)}
    (run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

