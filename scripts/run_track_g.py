"""Build, boot and compare xv6 round-robin and weighted-stride schedulers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import selectors
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "vendor/xv6-riscv-track-g-baseline"
STRIDE = ROOT / "vendor/xv6-riscv-track-g-stride"
PINNED = ROOT / "vendor/xv6-riscv-baseline"

RESULT_RE = re.compile(
    r"^BENCH_RESULT workload=(?P<workload>\S+) role=(?P<role>\d+) "
    r"weight=(?P<weight>\d+) elapsed=(?P<elapsed>\d+) "
    r"response=(?P<response>-?\d+) ticks=(?P<ticks>-?\d+) "
    r"dispatches=(?P<dispatches>-?\d+) checksum=(?P<checksum>-?\d+)$", re.M
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build(source: Path, out: Path, cpus: int) -> dict:
    detflags = shlex.quote(f"-ffile-prefix-map={source}=.")
    command = ["make", "-j4", "-B", "TOOLPREFIX=riscv64-elf-",
               f"DETFLAGS={detflags}", f"CPUS={cpus}", "kernel/kernel", "fs.img"]
    started = time.monotonic()
    with out.open("w") as log:
        result = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT)
    return {"command": command, "returncode": result.returncode,
            "seconds": time.monotonic() - started}


def qemu_command(cpus: int) -> list[str]:
    return ["qemu-system-riscv64", "-machine", "virt", "-bios", "none",
            "-kernel", "kernel/kernel", "-m", "128M", "-smp", str(cpus),
            "-nographic", "-global", "virtio-mmio.force-legacy=false",
            "-drive", "file=fs.img,if=none,format=raw,id=x0",
            "-device", "virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0"]


def run_guest(policy: str, source: Path, out: Path, cpus: int, repetitions: int,
              run_usertests: bool, timeout: int) -> dict:
    qemu = qemu_command(cpus)
    proc = subprocess.Popen(qemu, cwd=source, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    transcript = bytearray()
    commands = [workload for _ in range(repetitions)
                for workload in ("cpu", "mixed", "latency")]
    sent = 0
    completed = 0
    phase = "boot"
    phase_started = time.monotonic()
    total_started = phase_started
    tests_started = None
    metadata = {"policy": policy, "qemu_command": qemu, "guest_commands": [],
                "run_usertests": run_usertests, "result": "in_progress"}

    def send(command: str) -> None:
        metadata["guest_commands"].append(command)
        proc.stdin.write((command + "\n").encode())
        proc.stdin.flush()

    passed = False
    try:
        with out.open("wb") as log:
            while True:
                now = time.monotonic()
                phase_limit = timeout if phase == "tests" else 120
                if now - phase_started > phase_limit:
                    raise TimeoutError(f"{policy} timeout in {phase}")
                events = selector.select(timeout=1)
                if events:
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError(f"QEMU exited early: {proc.poll()}")
                    transcript.extend(chunk)
                    log.write(chunk)
                    log.flush()
                text = transcript.decode(errors="replace").replace("\r", "")
                if "panic:" in text or "BENCH_ERROR" in text or "SOME TESTS FAILED" in text:
                    raise RuntimeError(f"{policy} guest reported failure")
                if phase == "boot" and "init: starting sh" in text and "$ " in text:
                    phase = "benchmarks"
                    phase_started = now
                    send(f"schedbench {commands[0]}")
                    sent = 1
                elif phase == "benchmarks":
                    observed = len(re.findall(r"^BENCH_END workload=\S+\n", text, re.M))
                    if observed > completed:
                        completed = observed
                        phase_started = now
                        if sent < len(commands):
                            send(f"schedbench {commands[sent]}")
                            sent += 1
                        elif run_usertests:
                            phase = "tests"
                            tests_started = now
                            send("usertests")
                        else:
                            phase = "final_echo"
                            send(f"echo TRACK_G_{policy.upper()}_DONE")
                elif phase == "tests" and re.search(r"^ALL TESTS PASSED$", text, re.M):
                    metadata["usertests_seconds"] = now - tests_started
                    phase = "final_echo"
                    phase_started = now
                    send(f"echo TRACK_G_{policy.upper()}_DONE")
                elif phase == "final_echo" and re.search(
                        rf"^(?:\$ )?TRACK_G_{policy.upper()}_DONE$", text, re.M):
                    passed = True
                    metadata["result"] = "passed"
                    break
    except Exception as exc:
        metadata.update(result="failed", error=str(exc), phase_at_failure=phase)
    finally:
        selector.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        remaining = proc.stdout.read()
        if remaining:
            transcript.extend(remaining)
            with out.open("ab") as log:
                log.write(remaining)
        proc.stdout.close()
        proc.stdin.close()
    text = transcript.decode(errors="replace").replace("\r", "")
    raw = [match.groupdict() for match in RESULT_RE.finditer(text)]
    records = []
    seen: dict[tuple[str, int], int] = {}
    for item in raw:
        key = (item["workload"], int(item["role"]))
        repetition = seen.get(key, 0)
        seen[key] = repetition + 1
        records.append({"policy": policy, "repetition": repetition,
                        **{key: (int(value) if key != "workload" else value)
                           for key, value in item.items()}})
    metadata.update(
        passed=passed, benchmark_commands=len(commands), benchmark_completions=completed,
        benchmark_records=len(records), usertests_passed="ALL TESTS PASSED" in text,
        test_names_started=re.findall(r"test ([a-zA-Z0-9_]+):", text),
        console_sha256=sha256(out), qemu_seconds=time.monotonic() - total_started,
        qemu_exit_returncode=proc.returncode, qemu_stopped_by_harness=True,
    )
    if not passed:
        raise RuntimeError(f"{policy} guest failed: {metadata}")
    return {"metadata": metadata, "records": records}


def create_patch(destination: Path) -> dict:
    files = [
        "Makefile", "kernel/defs.h", "kernel/proc.c", "kernel/proc.h",
        "kernel/sched.h", "kernel/syscall.c", "kernel/syscall.h",
        "kernel/sysproc.c", "kernel/trap.c", "user/schedbench.c",
        "user/user.h", "user/usys.pl",
    ]
    chunks = []
    returncodes = []
    for relative in files:
        command = ["diff", "-uN", "--label", f"a/{relative}", "--label", f"b/{relative}",
                   str(PINNED / relative), str(STRIDE / relative)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode(errors="replace"))
        returncodes.append(result.returncode)
        if result.stdout:
            chunks.append(result.stdout)
    payload = b"".join(chunks)
    destination.write_bytes(payload)
    return {"method": "diff -uN over explicit source allowlist", "source_files": files,
            "returncodes": returncodes, "bytes": len(payload),
            "sha256": sha256(destination)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--skip-usertests", action="store_true")
    args = parser.parse_args()
    if args.repetitions <= 0 or args.timeout <= 0:
        parser.error("repetitions and timeout must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = ROOT / "results/G_stride" / stamp
    out.mkdir(parents=True, exist_ok=False)
    tools = {}
    for tool in ("qemu-system-riscv64", "riscv64-elf-gcc", "make", "diff"):
        if not shutil.which(tool):
            raise RuntimeError(f"missing tool {tool}")
        tools[tool] = subprocess.check_output([tool, "--version"], text=True,
                                              stderr=subprocess.STDOUT).splitlines()[0]
    metadata = {
        "step": "07", "run_kind": "debug" if args.skip_usertests else "final",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": json.loads((ROOT / "vendor/xv6-source.json").read_text())["commit"],
        "host": platform.platform(), "emulated_cpus": args.cpus, "memory_mib": 128,
        "timer_resolution": "one xv6 tick; timer programmed for ~0.1 seconds",
        "repetitions": args.repetitions, "tools": tools, "status": "running",
        "provenance": {"git_managed_by": "user", "git_revision": "pending user Git work"},
    }
    write_json(out / "metadata.json", metadata)
    try:
        builds = {}
        for policy, source in (("baseline", BASE), ("stride", STRIDE)):
            builds[policy] = build(source, out / f"build_{policy}.log", args.cpus)
            if builds[policy]["returncode"]:
                raise RuntimeError(f"{policy} build failed")
            builds[policy]["kernel_sha256"] = sha256(source / "kernel/kernel")
            builds[policy]["initial_fs_sha256"] = sha256(source / "fs.img")
        baseline = run_guest("baseline", BASE, out / "console_baseline.log", args.cpus,
                             args.repetitions, False, args.timeout)
        stride = run_guest("stride", STRIDE, out / "console_stride.log", args.cpus,
                           args.repetitions, not args.skip_usertests, args.timeout)
        records = baseline["records"] + stride["records"]
        expected_per_policy = args.repetitions * (3 + 2 + 3)
        if len(baseline["records"]) != expected_per_policy or len(stride["records"]) != expected_per_policy:
            raise AssertionError("unexpected benchmark result count")
        checksum_map = {}
        for record in records:
            key = (record["workload"], record["role"], record["repetition"])
            checksum_map.setdefault(key, set()).add(record["checksum"])
        if any(len(values) != 1 for values in checksum_map.values()):
            raise AssertionError("baseline/stride checksum mismatch")
        patch = create_patch(out / "xv6-stride.patch")
        (out / "records.json").write_text(json.dumps(records, indent=2) + "\n")
        summary = {"record_count": len(records), "records_per_policy": expected_per_policy,
                   "builds": builds, "guest": {"baseline": baseline["metadata"],
                                                "stride": stride["metadata"]},
                   "checksums_identical": True, "patch": patch}
        write_json(out / "summary.json", summary)
        metadata.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        artifacts={name: {"sha256": sha256(out / name)} for name in
                                   ("records.json", "summary.json", "console_baseline.log",
                                    "console_stride.log", "xv6-stride.patch")})
        write_json(out / "metadata.json", metadata)
        print(out)
        return 0
    except Exception as exc:
        metadata.update(status="failed", error=repr(exc),
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "metadata.json", metadata)
        print(f"FAILED: {exc}\nEvidence: {out}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
