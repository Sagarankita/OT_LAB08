"""Small, auditable benchmark-result infrastructure.

The module deliberately avoids Git subprocess calls because repository history is
managed by the user. Each run stores configuration and source hashes, raw records,
environment metadata, and an explicit status so incomplete/failed work cannot be
loaded as a completed result.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1
TERMINAL_STATUSES = {"completed", "failed", "invalid"}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class BenchmarkConfig:
    experiment: str
    run_kind: str
    seeds: tuple[int, ...]
    repetitions: int
    warmup_repetitions: int
    input_size: int
    timeout_seconds: float
    units: Mapping[str, str]
    solver_settings: Mapping[str, Any]
    force_failure: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BenchmarkConfig":
        required = {
            "experiment", "run_kind", "seeds", "repetitions", "warmup_repetitions", "input_size",
            "timeout_seconds", "units", "solver_settings",
        }
        missing = sorted(required - set(raw))
        extra = sorted(set(raw) - required - {"force_failure"})
        if missing:
            raise ValueError(f"missing configuration keys: {missing}")
        if extra:
            raise ValueError(f"unknown configuration keys: {extra}")
        experiment = raw["experiment"]
        run_kind = raw["run_kind"]
        if not isinstance(experiment, str) or not experiment.strip():
            raise ValueError("experiment must be a non-empty string")
        if run_kind not in {"debug", "final", "failure_probe"}:
            raise ValueError("run_kind must be debug, final, or failure_probe")
        seeds_raw = raw["seeds"]
        if not isinstance(seeds_raw, list) or not seeds_raw:
            raise ValueError("seeds must be a non-empty list")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
               for seed in seeds_raw):
            raise ValueError("each seed must be a non-negative integer")
        if len(set(seeds_raw)) != len(seeds_raw):
            raise ValueError("seeds must be unique")
        timeout = raw["timeout_seconds"]
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        units = raw["units"]
        solver_settings = raw["solver_settings"]
        if not isinstance(units, dict) or not units or not all(
            isinstance(k, str) and isinstance(v, str) and v for k, v in units.items()
        ):
            raise ValueError("units must be a non-empty string-to-string object")
        if not isinstance(solver_settings, dict):
            raise ValueError("solver_settings must be an object")
        force_failure = raw.get("force_failure", False)
        if not isinstance(force_failure, bool):
            raise ValueError("force_failure must be boolean")
        return cls(
            experiment=experiment.strip(), run_kind=run_kind,
            seeds=tuple(seeds_raw), repetitions=_positive_int("repetitions", raw["repetitions"]),
            warmup_repetitions=_nonnegative_int("warmup_repetitions", raw["warmup_repetitions"]),
            input_size=_positive_int("input_size", raw["input_size"]),
            timeout_seconds=float(timeout), units=dict(units),
            solver_settings=dict(solver_settings), force_failure=force_failure,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment, "run_kind": self.run_kind,
            "seeds": list(self.seeds), "repetitions": self.repetitions,
            "warmup_repetitions": self.warmup_repetitions,
            "input_size": self.input_size, "timeout_seconds": self.timeout_seconds,
            "units": dict(self.units), "solver_settings": dict(self.solver_settings),
            "force_failure": self.force_failure,
        }


def dependency_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


class RunRecorder:
    """Record one run and enforce a single explicit terminal state."""

    def __init__(self, results_root: Path, config_path: Path, config: BenchmarkConfig,
                 source_files: Iterable[Path], command: list[str]) -> None:
        self.config = config
        self.started_at = utc_now()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.directory = results_root / config.experiment / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self._status = "running"
        self._records: list[dict[str, Any]] = []
        config_bytes = canonical_json_bytes(config.as_dict())
        (self.directory / "config.json").write_bytes(config_bytes)
        source_hashes = {str(path): sha256_file(path) for path in sorted(source_files)}
        self.metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "experiment": config.experiment,
            "run_kind": config.run_kind,
            "started_at_utc": self.started_at,
            "command": command,
            "cwd": str(Path.cwd()),
            "config_source": str(config_path),
            "config_sha256": sha256_bytes(config_bytes),
            "source_sha256": source_hashes,
            "environment": {
                "python": sys.version,
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "logical_cpus": os.cpu_count(),
                "dependencies": dependency_versions(
                    ["numpy", "scipy", "matplotlib", "pytest", "torch", "pennylane", "coverage"]
                ),
            },
            "provenance": {
                "git_managed_by": "user",
                "git_revision": "pending user Git work",
                "working_tree_state": "not inspected; no Git command permitted by user instruction",
            },
            "units": dict(config.units),
            "solver_settings": dict(config.solver_settings),
        }
        _atomic_json(self.directory / "metadata.json", self.metadata)
        self._write_status("running")

    def _write_status(self, status: str, **details: Any) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION, "status": status,
            "started_at_utc": self.started_at,
            "updated_at_utc": utc_now(), **details,
        }
        _atomic_json(self.directory / "status.json", payload)

    def add_record(self, record: Mapping[str, Any]) -> None:
        if self._status != "running":
            raise RuntimeError("cannot append records to a terminal run")
        self._records.append(dict(record))

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self._records)

    def _save_records(self) -> None:
        _atomic_json(self.directory / "records.json", self._records)
        if not self._records:
            return
        fields = sorted({key for record in self._records for key in record})
        with (self.directory / "records.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._records)

    def complete(self, summary: Mapping[str, Any], artifacts: Mapping[str, Path]) -> None:
        if self._status != "running":
            raise RuntimeError("run already has a terminal status")
        self._save_records()
        artifact_hashes = {}
        for name, path in artifacts.items():
            resolved = path.resolve()
            if not resolved.is_file() or self.directory.resolve() not in resolved.parents:
                raise ValueError(f"artifact must be a file inside run directory: {name}")
            artifact_hashes[name] = {
                "path": str(path.relative_to(self.directory)), "sha256": sha256_file(path),
            }
        _atomic_json(self.directory / "summary.json", dict(summary))
        self._status = "completed"
        self._write_status("completed", record_count=len(self._records), artifacts=artifact_hashes)

    def fail(self, exc: BaseException) -> None:
        if self._status != "running":
            return
        self._save_records()
        error = {
            "type": type(exc).__name__, "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        }
        _atomic_json(self.directory / "error.json", error)
        self._status = "failed"
        self._write_status("failed", record_count=len(self._records), error_file="error.json")


def record_invalid_attempt(results_root: Path, config_path: Path, raw_bytes: bytes,
                           exc: BaseException, command: list[str]) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = results_root / "invalid" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "input.json").write_bytes(raw_bytes)
    _atomic_json(directory / "error.json", {
        "type": type(exc).__name__, "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    })
    _atomic_json(directory / "status.json", {
        "schema_version": SCHEMA_VERSION, "status": "invalid",
        "updated_at_utc": utc_now(), "config_source": str(config_path),
        "input_sha256": sha256_bytes(raw_bytes), "command": command,
        "provenance": {"git_revision": "pending user Git work", "git_command_run": False},
    })
    return directory


def numeric_summary(records: Iterable[Mapping[str, Any]], group_key: str,
                    value_key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(str(record[group_key]), []).append(float(record[value_key]))
    return {
        group: {
            "count": len(values), "mean": fmean(values),
            "sample_std": stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values), "maximum": max(values),
        }
        for group, values in sorted(grouped.items())
    }


def load_completed_run(directory: Path) -> dict[str, Any]:
    status = json.loads((directory / "status.json").read_text())
    if status.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported result schema")
    if status.get("status") != "completed":
        raise ValueError(f"run is not completed: {status.get('status')}")
    required = ["config.json", "metadata.json", "records.json", "summary.json"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"completed run is missing files: {missing}")
    return {name[:-5]: json.loads((directory / name).read_text()) for name in required}

