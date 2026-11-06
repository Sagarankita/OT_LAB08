#!/usr/bin/env python3
"""Profile one more-itertools test selection with per-test coverage contexts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import coverage
import pytest


class ProfilePlugin:
    def __init__(self, cov: coverage.Coverage) -> None:
        self.cov = cov
        self.durations: dict[str, float] = {}
        self.outcomes: dict[str, str] = {}

    def pytest_runtest_setup(self, item: pytest.Item) -> None:
        self.cov.switch_context(item.nodeid)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.durations[report.nodeid] = self.durations.get(report.nodeid, 0.0) + report.duration
        if report.when == "call" or (report.when == "setup" and report.failed):
            self.outcomes[report.nodeid] = report.outcome


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebase", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()

    codebase = args.codebase.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    nodeids = None
    if args.selection:
        nodeids = json.loads(args.selection.read_text(encoding="utf-8"))

    os.chdir(codebase)
    cov = coverage.Coverage(data_file=str(output.with_suffix(".coverage")), source=["more_itertools"])
    plugin = ProfilePlugin(cov)
    cov.start()
    cov.switch_context("__session__")
    pytest_args = [
        "-q",
        "--disable-warnings",
        "--rootdir=.",
        *(nodeids if nodeids is not None else ["tests"]),
    ]
    exit_code = int(pytest.main(pytest_args, plugins=[plugin]))
    cov.stop()
    cov.save()

    data = cov.get_data()
    file_lines: dict[str, list[int]] = {}
    per_test: dict[str, set[str]] = {test: set() for test in plugin.durations}
    session_lines: set[str] = set()
    source_root = codebase / "more_itertools"
    for filename in sorted(data.measured_files()):
        path = Path(filename).resolve()
        if source_root not in path.parents:
            continue
        relative = path.relative_to(codebase).as_posix()
        lines = sorted(data.lines(filename) or [])
        file_lines[relative] = lines
        contexts = data.contexts_by_lineno(filename)
        for line in lines:
            key = f"{relative}:{line}"
            line_contexts = contexts.get(line, [])
            if "__session__" in line_contexts:
                session_lines.add(key)
            for context in line_contexts:
                if context in per_test:
                    per_test[context].add(key)

    payload = {
        "exit_code": exit_code,
        "pytest_args": pytest_args,
        "durations_seconds": plugin.durations,
        "outcomes": plugin.outcomes,
        "file_lines": file_lines,
        "executed_lines": sorted(
            f"{filename}:{line}" for filename, lines in file_lines.items() for line in lines
        ),
        "session_lines": sorted(session_lines),
        "per_test_coverage": {test: sorted(lines) for test, lines in sorted(per_test.items())},
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
