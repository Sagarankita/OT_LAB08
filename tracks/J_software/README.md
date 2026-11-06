# Track J — coverage-preserving test-suite selection

## Process and pinned codebase

This track optimizes repeated CI test execution for `more-itertools` 10.7.0, a public MIT-licensed Python project. The unmodified source distribution is pinned by SHA-256 `9fddd5403be01a94b204faadcff459ec3568cf110265d3c54323e1e866ad29d3`. Its three production Python files contain 6,353 physical lines and 4,728 nonblank, non-comment physical lines, comfortably exceeding the 1,000-line requirement. The selected final evidence is in `results/track_j_test_selection/20261028T090927.339437Z/`.

An earlier candidate, `cachetools` 5.5.2, was rejected because its 1,073 physical source lines could fall below 1,000 after meaningful-line filtering. It is not the experimental subject.

## Formulation

Let test `i` have measured duration `t_i` and let `a_li=1` when its full-profile coverage context executes target production line `l`. Choose binary variables `x_i`:

```text
minimize    sum_i t_i x_i
subject to  sum_i a_li x_i >= 1    for every selectable target line l
            x_i in {0, 1}
```

The full suite executed 2,030 production lines. Of these, 308 import/session lines occur independently of an individual test and 1,722 lines form the selection constraints. Every reduced method uses this identical target. The objective is the sum of per-test pytest durations; fresh-process wall time is measured separately because interpreter and collection overhead are shared costs.

Methods:

- Full baseline: all 671 tests.
- Naive: lexical first-cover selection.
- Greedy: maximum newly covered lines per measured second.
- ILP: binary weighted set cover solved by SciPy/HiGHS with zero relative MIP gap; the final solve was optimal.

## Fair measurement and results

The profiler uses coverage.py contexts to collect per-test line coverage and pytest report durations in one full-suite run. Each chosen suite is then rerun under coverage to check the constraint, followed by one warm-up and three fresh-process measured executions. All methods use the same Python, source, pytest options and target lines.

| Method | Tests | Mean wall time (s) | SD (s) | Reduction | Selection overhead (s) | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Full | 671 | 6.999 | 0.009 | 0.00% | 0 | 2,030/2,030 |
| Naive | 304 | 5.682 | 0.025 | 18.81% | 0.00048 | 2,030/2,030 |
| Greedy | 263 | 5.399 | 0.005 | 22.86% | 0.03231 | 2,030/2,030 |
| ILP | 242 | 5.392 | 0.015 | 22.96% | 0.00940 | 2,030/2,030 |

The complete initial plus three verification profiles cost 92.60 seconds (full 27.98, naive 21.91, greedy 21.28 and ILP 21.43 seconds). Relative to the full suite, the ILP suite saves about 1.607 seconds per CI execution, so this complete experimental profiling cost would amortize after roughly 58 repeated runs if coverage relationships remain stable. In a production workflow only the initial/full profile and periodic refresh are needed; this experiment deliberately re-profiled every alternative for correctness.

The smaller ILP test count did not translate into a large advantage over greedy wall time because process startup, collection and a few expensive selected tests dominate. This is why the report does not equate percentage test-count reduction with percentage runtime reduction.

## Correctness and fault-detection check

- Every selected suite passed and independently re-executed all 2,030 target lines.
- The verifier reproduced naive, greedy and optimal ILP selections from the raw full profile and passed 22 checks.
- Two deterministic temporary mutants (`take` off-by-one and `quantify` plus one) were killed by the full, naive, greedy and ILP suites: 8/8 mutation-method checks.
- The source archive hash, license, source size, benchmark sample counts and plot were independently checked.

The mutation sample is intentionally small. Equal line coverage and killing two mutants do not prove equal fault detection. Branch/value coverage, correlated tests, flaky paths and future code changes can invalidate a selection.

## Failures, maintenance and limitations

The first profiler used node IDs relative to the parent repository, which broke selected-suite replay after changing directories; pinning `--rootdir=.` fixed it. One thread-based test also showed timing-dependent coverage in a debug run, motivating replay validation and a deterministic repair mechanism (no repair was needed in the selected final run). The initial plotting attempt used an interactive macOS backend and exited before saving the summary; `MPLBACKEND=Agg` fixed it. These are recorded in `docs/AI_FAILURES.md`.

Test selection must be regenerated when tests, production code, Python behavior or coverage tooling changes. It also reduces redundancy that may catch faults through different values even when line coverage is unchanged. A maintainable CI design should retain the full suite periodically and on high-risk changes, cache profiles by pinned revision, and use reduced suites only where the risk/cost trade-off is acceptable.

## Reproduction

```sh
.venv/bin/python scripts/run_track_j.py --config configs/track_j_debug.json
.venv/bin/python scripts/run_track_j.py --config configs/track_j_final.json
.venv/bin/python scripts/verify_track_j.py results/track_j_test_selection/20261028T090927.339437Z
.venv/bin/python -m pytest
```

Git commits and publication remain user-managed.
