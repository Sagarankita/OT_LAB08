import json
from pathlib import Path

import pytest

from optimization_lab.benchmarking import BenchmarkConfig, load_completed_run, numeric_summary


def valid_config():
    return {
        "experiment": "unit", "run_kind": "debug", "seeds": [1],
        "repetitions": 1, "warmup_repetitions": 1,
        "input_size": 8, "timeout_seconds": 1,
        "units": {"time": "seconds"}, "solver_settings": {},
    }


@pytest.mark.parametrize("field,value", [
    ("seeds", []), ("seeds", [1, 1]), ("repetitions", 0),
    ("warmup_repetitions", -1),
    ("input_size", -2), ("timeout_seconds", 0), ("units", {}),
])
def test_rejects_invalid_config(field, value):
    raw = valid_config()
    raw[field] = value
    with pytest.raises(ValueError):
        BenchmarkConfig.from_mapping(raw)


def test_rejects_unknown_keys():
    raw = valid_config()
    raw["typo"] = True
    with pytest.raises(ValueError, match="unknown"):
        BenchmarkConfig.from_mapping(raw)


def test_numeric_summary_reports_sample_variability():
    summary = numeric_summary([
        {"method": "m", "time": 1.0}, {"method": "m", "time": 3.0}
    ], "method", "time")
    assert summary["m"] == {
        "count": 2, "mean": 2.0, "sample_std": pytest.approx(2 ** 0.5),
        "minimum": 1.0, "maximum": 3.0,
    }


@pytest.mark.parametrize("status", ["running", "failed", "invalid"])
def test_loader_rejects_noncompleted_run(tmp_path: Path, status: str):
    (tmp_path / "status.json").write_text(json.dumps({"schema_version": 1, "status": status}))
    with pytest.raises(ValueError, match="not completed"):
        load_completed_run(tmp_path)
