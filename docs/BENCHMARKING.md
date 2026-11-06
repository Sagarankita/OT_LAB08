# Shared benchmark and result format

Prompt 02 created this infrastructure. Its smoke timings test the machinery only; they are not evidence that an Assignment 8 optimization improved a process.

## Run lifecycle

Each invocation creates a new timestamped directory. A run begins with `status: running` and ends exactly once as `completed`, `failed`, or `invalid`.

- `completed`: validated configuration, raw records, summary, metadata and declared artifacts exist. Only these runs can be loaded by `load_completed_run`.
- `failed`: a validated run raised an exception. Partial records and an `error.json` traceback are retained.
- `invalid`: configuration could not be validated. The exact input bytes, their hash and validation error are retained separately.

An expected failure probe exists only to verify this lifecycle. It is not reported as an AI-generated failure or an experiment result.

## Evidence in a completed run

- `config.json`: validated, canonical configuration.
- `metadata.json`: command, environment and dependency versions, input/config/source hashes, units, solver settings, timing policy and provenance.
- `records.json` and `records.csv`: individual observations, including seed and repetition.
- `summary.json`: aggregate count, mean, sample standard deviation, minimum and maximum.
- `status.json`: terminal state, record count and artifact hashes.
- Plot files: generated from the saved observations.
- `verification.json`: independent reload and consistency checks, created by a separate verifier.

Historical result metadata records `pending user Git work` because the experiments predated repository initialization. The verified project snapshot was later committed as `9757572` and published at `https://github.com/Sagarankita/OT_LAB08`; measured artifacts were not rewritten retroactively.

## Reproduce Prompt 02

From the project root:

```sh
.venv/bin/python -m pytest
.venv/bin/python scripts/run_infrastructure_smoke.py --config configs/smoke_debug.json
.venv/bin/python scripts/run_infrastructure_smoke.py --config configs/smoke_invalid.json
.venv/bin/python scripts/run_infrastructure_smoke.py --config configs/smoke_failure.json
.venv/bin/python scripts/run_infrastructure_smoke.py --config configs/smoke_final.json
.venv/bin/python scripts/verify_infrastructure_smoke.py results/infrastructure_smoke/<final-run-id>
```

The invalid and failure commands intentionally exit with status 2 and 1 respectively. Automation must expect those exact nonzero outcomes and inspect their saved status files rather than treating them as missing evidence.

## Verified smoke result

The selected final run is `results/infrastructure_smoke/20261027T133503.973784Z`.

- Five fixed seeds, five measured repetitions per seed and method, and two unmeasured warm-ups per seed and method.
- 50 raw records: 5 seeds × 5 repetitions × 2 methods.
- Input arrays have distinct SHA-256 hashes, and both implementations agree with the `math.fsum` reference within the recorded tolerance.
- Setup/warm-up time is separate from measured execution time.
- Units, solver/method settings, package versions, source/config/input hashes and user-managed Git provenance are present.
- The independent verifier passed all 15 checks and verified the saved plot hash.

The first final smoke run, `20261027T133430.920509Z`, is retained as superseded evidence. Visual review found that first-call overhead inflated one method's variation and allowed a symmetric time error bar to extend below zero. The corrected run adds explicit warm-ups to setup and constrains the time axis to non-negative values. See `docs/AI_FAILURES.md`.

## Rules for later tracks

Use the same inputs, stopping criteria, resource limits and measurement definitions for methods being compared. Start with a debug configuration, then freeze a final configuration. Preserve individual stochastic runs; report variability rather than only the best seed. Keep setup, compilation, model loading and index construction separate from execution when the distinction matters, while reporting both costs. Preserve solver termination state, bound and gap; a feasible timeout is not an optimum. Add track-specific validators rather than relying only on successful process exit.

