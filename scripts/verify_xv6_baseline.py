"""Independently audit a saved successful baseline run against pinned source."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    metadata = json.loads((run / "metadata.json").read_text())
    pin = json.loads((ROOT / "vendor/xv6-source.json").read_text())
    manifest = json.loads((ROOT / "vendor/xv6-source-manifest.json").read_text())
    source = ROOT / pin["source_directory"]
    for name, digest in manifest.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Source differs: {name}")
    raw = (run / "console.log").read_bytes()
    console = raw.decode("utf-8", "replace").replace("\r", "")
    text = (source / "user/usertests.c").read_text()
    # Remove commented-out cases; extract the two actual default test arrays.
    text = re.sub(r"//[^\n]*", "", text)
    expected = []
    for array in ("quicktests", "slowtests"):
        match = re.search(r"\b" + array + r"\[\]\s*=\s*\{(.*?)\n\};", text, re.S)
        if not match:
            raise SystemExit(f"Cannot locate upstream {array}")
        expected.extend(re.findall(r'\{\s*\w+\s*,\s*"([^"]+)"\s*\}', match.group(1)))
    observed = re.findall(r"test ([a-zA-Z0-9_]+):", console)
    checks = {
        "metadata_reports_success": metadata["result"] == "passed",
        "build_succeeded": metadata["build_returncode"] == 0,
        "upstream_commit_matches_pin": metadata["upstream_commit"] == pin["commit"],
        "console_hash_matches": hashlib.sha256(raw).hexdigest() == metadata["console_sha256"],
        # Early saved runs sent the next command as soon as the marker bytes
        # arrived, before echo's separate newline write. The terminal then
        # appended the known input echo. This is real guest marker output,
        # not the line "echo BASELINE_BOOT_OK". Accept only that exact known
        # command echo suffix, not arbitrary extra text.
        "boot_echo": bool(re.search(r"^(?:\$ )?BASELINE_BOOT_OK(?:usertests)?$", console, re.M)),
        "full_usertests_command": metadata["guest_commands"] == [
            "echo BASELINE_BOOT_OK", "usertests", "echo BASELINE_POST_TEST_OK"],
        "all_expected_tests_started_in_order": bool(expected) and observed == expected,
        "all_tests_passed_marker": bool(re.search(r"^ALL TESTS PASSED$", console, re.M)),
        "post_test_shell_echo": bool(re.search(r"^(?:\$ )?BASELINE_POST_TEST_OK$", console, re.M)),
        "no_failure_or_panic": not re.search(r"\bFAILED\b|panic:", console),
        "kernel_hash_matches": hashlib.sha256((source / "kernel/kernel").read_bytes()).hexdigest()
                               == metadata["kernel_sha256"],
    }
    report = {"checks": checks, "expected_tests": len(expected), "observed_tests": len(observed),
              "original_source_files_verified": len(manifest), "passed": all(checks.values())}
    (run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
