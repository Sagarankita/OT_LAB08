"""Build pinned unmodified xv6, boot QEMU and run full usertests.

No Git commands. Each invocation creates a new evidence directory. The QEMU
process uses an isolated, freshly built disk image and is stopped after the
post-test shell responds. Host runtime is setup/test evidence, not a speedup.
"""
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


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(source):
    manifest = json.loads((ROOT / "vendor/xv6-source-manifest.json").read_text())
    changed = [name for name, digest in manifest.items()
               if not (source / name).is_file() or file_hash(source / name) != digest]
    if changed:
        raise RuntimeError(f"Baseline original sources differ: {changed}")
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    source = ROOT / "vendor/xv6-riscv-baseline"
    pin = json.loads((ROOT / "vendor/xv6-source.json").read_text())
    verified = verify_source(source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = ROOT / "results/G_baseline" / stamp
    out.mkdir(parents=True, exist_ok=False)
    tools = ["qemu-system-riscv64", "riscv64-elf-gcc", "riscv64-elf-ld", "make"]
    versions = {}
    for tool in tools:
        if not shutil.which(tool):
            raise RuntimeError(f"Missing tool: {tool}")
        versions[tool] = subprocess.check_output([tool, "--version"], text=True).splitlines()[0]
    # Make invokes a shell. Quote the entire compiler flag so spaces in CURDIR
    # do not split -ffile-prefix-map. This changes no upstream source file.
    detflags = shlex.quote(f"-ffile-prefix-map={source}=.")
    build = ["make", "-j4", "-B", "TOOLPREFIX=riscv64-elf-",
             f"DETFLAGS={detflags}", f"CPUS={args.cpus}", "kernel/kernel", "fs.img"]
    qemu = ["qemu-system-riscv64", "-machine", "virt", "-bios", "none",
            "-kernel", "kernel/kernel", "-m", "128M", "-smp", str(args.cpus),
            "-nographic", "-global", "virtio-mmio.force-legacy=false",
            "-drive", "file=fs.img,if=none,format=raw,id=x0",
            "-device", "virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0"]
    metadata = {
        "step": "01", "purpose": "unmodified xv6 setup and full usertests baseline",
        "upstream_commit": pin["commit"], "original_sources_verified": verified,
        "host_platform": platform.platform(), "host_logical_cpus": os.cpu_count(),
        "emulated_cpus": args.cpus, "memory_mib": 128, "tools": versions,
        "cwd": str(source.relative_to(ROOT)), "build_command": build,
        "qemu_command": qemu, "guest_commands": [], "timeout_seconds": args.timeout,
        "result": "in_progress", "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    def save():
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save()
    start = time.monotonic()
    with (out / "build.log").open("w") as log:
        result = subprocess.run(build, cwd=source, stdout=log, stderr=subprocess.STDOUT)
    metadata.update(build_returncode=result.returncode, build_seconds=time.monotonic() - start)
    save()
    if result.returncode:
        metadata["result"] = "build_failed"
        save()
        raise SystemExit(f"Build failed; see {out / 'build.log'}")
    metadata["kernel_sha256"] = file_hash(source / "kernel/kernel")
    metadata["initial_fs_sha256"] = file_hash(source / "fs.img")
    print(f"Build passed. Evidence: {out.relative_to(ROOT)}", flush=True)
    proc = subprocess.Popen(qemu, cwd=source, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    transcript = bytearray()
    phase = "boot"
    runtime_start = time.monotonic()
    phase_start = runtime_start
    last_progress = runtime_start
    passed = False
    try:
        with (out / "console.log").open("wb") as log:
            while True:
                now = time.monotonic()
                limit = args.timeout if phase == "tests" else 45
                if now - phase_start > limit:
                    raise TimeoutError(f"Timeout in phase {phase} after {limit}s")
                events = selector.select(timeout=1)
                if events:
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError(f"QEMU exited before completion: {proc.poll()}")
                    log.write(chunk)
                    log.flush()
                    transcript.extend(chunk)
                text = transcript.decode("utf-8", errors="replace").replace("\r", "")
                if "panic:" in text or "SOME TESTS FAILED" in text or re.search(r"\bFAILED\b", text):
                    raise RuntimeError("Guest reported panic or test failure; inspect console.log")
                command = None
                if phase == "boot" and "init: starting sh" in text and "$ " in text:
                    command, phase = "echo BASELINE_BOOT_OK", "boot_echo"
                    metadata["boot_seconds"] = now - runtime_start
                elif phase == "boot_echo" and re.search(r"^(?:\$ )?BASELINE_BOOT_OK$", text, re.M):
                    command, phase = "usertests", "tests"
                    metadata["tests_started_at_utc"] = datetime.now(timezone.utc).isoformat()
                elif phase == "tests" and re.search(r"^ALL TESTS PASSED$", text, re.M):
                    metadata["usertests_seconds"] = now - phase_start
                    command, phase = "echo BASELINE_POST_TEST_OK", "post_test"
                elif phase == "post_test" and re.search(r"^(?:\$ )?BASELINE_POST_TEST_OK$", text, re.M):
                    passed = True
                    metadata["result"] = "passed"
                    break
                if command:
                    metadata["guest_commands"].append(command)
                    proc.stdin.write((command + "\n").encode())
                    proc.stdin.flush()
                    phase_start = now
                    save()
                    print(f"Guest command: {command}", flush=True)
                if now - last_progress >= 20:
                    tests = re.findall(r"test ([a-zA-Z0-9_]+):", text)
                    print(f"Phase={phase}; elapsed={now-runtime_start:.1f}s; "
                          f"last test={tests[-1] if tests else 'none'}", flush=True)
                    last_progress = now
    except Exception as exc:
        metadata.update(result="failed", error=str(exc), phase_at_failure=phase)
        print(f"FAILED: {exc}", flush=True)
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
            with (out / "console.log").open("ab") as log:
                log.write(remaining)
        proc.stdout.close()
        proc.stdin.close()
        metadata.update(
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            qemu_total_seconds=time.monotonic() - runtime_start,
            qemu_exit_returncode=proc.returncode,
            qemu_stopped_by_harness=True,
            original_sources_verified_after=verify_source(source),
            console_sha256=file_hash(out / "console.log"),
        )
        text = transcript.decode("utf-8", errors="replace")
        metadata["test_names_started"] = re.findall(r"test ([a-zA-Z0-9_]+):", text)
        save()
    print(json.dumps(metadata, indent=2), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
