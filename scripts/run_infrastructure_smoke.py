"""Run the setup-only benchmark smoke test and save auditable evidence."""
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
import numpy as np

from optimization_lab.benchmarking import (
    BenchmarkConfig, RunRecorder, canonical_json_bytes, numeric_summary,
    record_invalid_attempt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
               "--results-root", str(args.results_root.resolve())]
    try:
        raw_bytes = config_path.read_bytes()
        raw = json.loads(raw_bytes)
        config = BenchmarkConfig.from_mapping(raw)
    except Exception as exc:
        raw_bytes = config_path.read_bytes() if config_path.is_file() else b""
        directory = record_invalid_attempt(args.results_root.resolve(), config_path,
                                           raw_bytes, exc, command)
        print(f"INVALID: {exc}\nEvidence: {directory}", file=sys.stderr)
        return 2

    recorder = RunRecorder(
        args.results_root.resolve(), config_path, config,
        source_files=[Path(__file__).resolve(), ROOT / "src/optimization_lab/benchmarking.py"],
        command=command,
    )
    setup_started = time.perf_counter()
    try:
        if config.force_failure:
            raise RuntimeError("intentional failure probe requested by validated configuration")
        inputs: dict[int, np.ndarray] = {}
        input_hashes: dict[str, str] = {}
        references: dict[int, float] = {}
        for seed in config.seeds:
            data = np.random.default_rng(seed).normal(size=config.input_size).astype(np.float64)
            inputs[seed] = data
            input_hashes[str(seed)] = hashlib.sha256(data.tobytes()).hexdigest()
            references[seed] = math.fsum(float(value) * float(value) for value in data)
        for seed in config.seeds:
            data = inputs[seed]
            for _ in range(config.warmup_repetitions):
                math.fsum(float(value) * float(value) for value in data)
                np.dot(data, data)
        setup_seconds = time.perf_counter() - setup_started
        recorder.metadata["input_sha256_by_seed"] = input_hashes
        recorder.metadata["timing_policy"] = {
            "setup_included_in_execution": False,
            "setup_seconds": setup_seconds,
            "clock": "time.perf_counter",
            "warmup_repetitions_per_seed_and_method": config.warmup_repetitions,
            "repetitions_per_seed_and_method": config.repetitions,
        }
        (recorder.directory / "metadata.json").write_bytes(canonical_json_bytes(recorder.metadata))
        deadline = time.perf_counter() + config.timeout_seconds
        for seed in config.seeds:
            data = inputs[seed]
            reference = references[seed]
            for repetition in range(config.repetitions):
                for method in ("python_fsum", "numpy_dot"):
                    if time.perf_counter() > deadline:
                        raise TimeoutError(f"execution exceeded {config.timeout_seconds} seconds")
                    started = time.perf_counter()
                    if method == "python_fsum":
                        objective = math.fsum(float(value) * float(value) for value in data)
                    else:
                        objective = float(np.dot(data, data))
                    elapsed = time.perf_counter() - started
                    absolute_error = abs(objective - reference)
                    tolerance = 1e-10 * max(1.0, abs(reference))
                    if absolute_error > tolerance:
                        raise AssertionError(
                            f"{method} disagrees with reference for seed {seed}: {absolute_error}"
                        )
                    recorder.add_record({
                        "seed": seed, "repetition": repetition, "method": method,
                        "input_sha256": input_hashes[str(seed)], "input_size": config.input_size,
                        "objective": objective, "absolute_error": absolute_error,
                        "execution_seconds": elapsed, "valid": True,
                    })
        timing = numeric_summary(recorder.records, "method", "execution_seconds")
        errors = numeric_summary(recorder.records, "method", "absolute_error")
        figure = recorder.directory / "timing_by_method.png"
        labels = list(timing)
        means = [timing[label]["mean"] for label in labels]
        deviations = [timing[label]["sample_std"] for label in labels]
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.bar(labels, means, yerr=deviations, capsize=5)
        axis.set_ylabel("Execution time (seconds; mean ± sample SD)")
        axis.set_title(f"Infrastructure smoke ({config.run_kind}; not optimization evidence)")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure, dpi=140)
        plt.close(fig)
        summary = {
            "scope": "infrastructure smoke only; not an assignment optimization result",
            "status": "completed", "run_kind": config.run_kind,
            "seed_count": len(config.seeds), "seeds": list(config.seeds),
            "repetitions": config.repetitions,
            "warmup_repetitions": config.warmup_repetitions,
            "record_count": len(recorder.records),
            "setup_seconds": setup_seconds, "timing_by_method": timing,
            "absolute_error_by_method": errors,
        }
        recorder.complete(summary, {"timing_plot": figure})
        print(recorder.directory)
        return 0
    except Exception as exc:
        recorder.fail(exc)
        print(f"FAILED: {exc}\nEvidence: {recorder.directory}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

