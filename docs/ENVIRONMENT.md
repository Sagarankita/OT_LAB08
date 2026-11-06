# Environment — step 00

Recorded 2026-10-27, Asia/Kolkata. Setup checks only; no assignment performance results.

- macOS 26.5.2, arm64; 10 logical CPUs reported by Python.
- Installed memory: 17,179,869,184 bytes (16 GiB), read with authorized `sysctl -n hw.memsize` after sandbox denied the initial query.
- Available disk at inspection: approximately 256 GiB (`df -h .`).
- Default Python: 3.14.2. Selected existing `/opt/homebrew/bin/python3.11`: Python 3.11.15, isolated in `.venv`. Selection avoids depending on newest-Python wheel availability; installed wheels and smoke checks validate this actual environment.
- Git 2.50.1, Apple Clang 21.0.0 and make available. Homebrew available.
- QEMU RISC-V and RISC-V GCC toolchains absent from PATH. Installation/build/boot are Prompt 01 work, not completed here.
- Raw versions and smoke results: `results/setup/environment.json`. Resolved packages: `requirements-lock.txt`.

## Commands actually run

```sh
uname -a
sw_vers
sysctl -n hw.ncpu hw.memsize
sysctl -n hw.memsize
df -h .
command -v python3 git clang make qemu-system-riscv64 riscv64-unknown-elf-gcc riscv64-linux-gnu-gcc brew
python3 --version
git --version
clang --version
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install numpy scipy matplotlib pytest torch pennylane coverage
.venv/bin/python -m pip install --no-cache-dir numpy scipy matplotlib pytest torch pennylane coverage
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > requirements-lock.txt
.venv/bin/python scripts/check_environment.py
```

Initial pip failed on sandbox DNS; the authorized network retry installed dependencies. `pip check` reported no broken requirements. Initial plot generation warned about fontconfig cache permissions; the script now sets project-local Matplotlib and XDG cache paths. Final smoke output is saved separately. No system packages were installed.

## Stack decisions

NumPy/SciPy for numerical work; SciPy `milp` (HiGHS) for ILP without an additional solver install; matplotlib for saved plots; pytest and coverage for verification; PyTorch CPU for learning; PennyLane default.qubit/default.mixed for quantum simulation. SQLite is Python's built-in module. No GPU, cloud or quantum hardware access is assumed.

Official references consulted for setup:

- [Python virtual environments](https://docs.python.org/3/library/venv.html)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [PennyLane installation](https://docs.pennylane.ai/en/stable/development/guide/installation.html)
- [SciPy MILP interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)

## Git and provenance

The folder inherits `/Users/ankitasagar/Desktop` as Git root. Project-local `git init .` was denied; its approval request was rejected. No initialization, commit or push succeeded. The user subsequently instructed local-only work and will handle Git. Do not stage unrelated Desktop files. Record commit references when supplied by the user.

## Prompt 01 toolchain update (2026-10-27)

The initial missing-tool finding above is historical. An authorized `brew install qemu riscv64-elf-gcc` installed QEMU 11.1.2, RISC-V GCC 16.2.0, and binutils 2.47.20260726. Homebrew updated itself and installed/upgraded dependencies, including glib, pcre2, libunistring, libpng, CA certificates and OpenSSL. These host changes are not in requirements-lock.txt. Details and actual per-run command/version records are in tracks/G_xv6/README.md and results/G_baseline/. No Git commands were used for Prompt 01; official source was obtained through an API and archive download.
