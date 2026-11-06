# AI failure log

At least three genuine failures are required before submission. Actual cases are recorded below; setup restrictions are not code failures. Coverage is not yet complete, and the student should distinguish invocation issues from substantive implementation errors.

For each actual case record: track/date, original output, failing check, cause, corrected code, verification, commit reference (pending user Git work), and lesson.

## Setup issue 01 — build command mishandled spaces (2026-10-27)

Actual AI-generated first build invocation used `make` without overriding the upstream Makefile's unquoted path flag. GCC rejected the split path with `invalid argument '/Users/ankitasagar/Desktop/OT' to '-ffile-prefix-map'`. Evidence: results/G_baseline/build.log and build.json. Fixed by using `shlex.quote` for the entire deterministic-path compiler flag in the command-line DETFLAGS override. Later build/test evidence is in timestamped run directories. Lesson: subprocess argument lists do not prevent a Makefile recipe from splitting unquoted path strings. Original upstream files remain unchanged.

This is a real setup/invocation error; it is not a fabricated kernel bug and does not replace the requirement to document substantive track implementation failures as they occur.

## Track G harness issue 02 — shell prompt caused false failure (2026-10-27)

AI-generated `run_xv6_baseline.py` initially matched the post-test echo with `^BASELINE_POST_TEST_OK$`. QEMU output contained `$ BASELINE_POST_TEST_OK` because the next command arrived before the shell printed its prompt. The guest's full `usertests` passed, but the harness timed out and correctly preserved its own failed status rather than silently declaring success. Evidence: results/G_baseline/20261027T132513.958400Z/{console.log,metadata.json}.

Fix: accept an optional anchored shell prefix with `^(?:\$ )?BASELINE_POST_TEST_OK$`, likewise for the boot echo, while still rejecting the echoed command `echo BASELINE_POST_TEST_OK`. Direct regression checks covered both valid outputs and both command-echo variants. The full baseline was rerun after correction; a separate timestamped directory preserves the new result. The independent verifier uses the corrected output rule and checks the complete upstream test list, not merely one success string.

Lesson: distinguish guest failure from harness failure; interactive console formatting can interleave prompts and output. This is a genuine AI-generated test-harness bug, not an injected fault. Commit references pending user Git work.

## Shared benchmark issue 03 — cold-start outlier and impossible error-bar range (2026-10-27)

The first AI-generated final smoke benchmark measured NumPy without an explicit warm-up. Its first call took about 0.000371 seconds while later calls were near 0.000001 seconds, inflating the sample standard deviation. The bar chart used symmetric error bars and consequently extended below zero seconds, an impossible and potentially misleading visual range. Evidence: `results/infrastructure_smoke/20261027T133430.920509Z/{records.json,summary.json,timing_by_method.png}`.

Fix: add a validated `warmup_repetitions` setting, perform and record warm-ups as setup rather than execution, add a verifier check for the timing policy, and constrain the plot's time axis to start at zero. Twelve unit tests passed, and the corrected five-seed run `20261027T133503.973784Z` passed all 15 independent checks. Visual inspection confirmed the corrected axis. Both runs remain available, with `latest-success.json` selecting the corrected run.

Lesson: explicitly define cold-start/warm-up policy and visually inspect uncertainty displays; statistically valid symmetric intervals can imply physically invalid values. This was a real AI-generated benchmarking weakness discovered during verification, not a deliberately injected failure. Commit reference pending user Git work.

## Track E issue 04 — time-indexed MILP found no incumbent under limit (2026-10-27)

The first AI-generated Track E final configuration applied a one-second limit to a time-indexed MILP. On a 24-job, three-worker instance, HiGHS reached the limit before producing any feasible incumbent, so the runner correctly stopped instead of recording a schedule or pretending the solve was optimal. Evidence: `results/track_e_scheduling/20261027T135049.886462Z/{status.json,error.json}`. Direct probes at two, three and five seconds also returned no incumbent for that instance.

Fix: replace the measured large-instance model with an equivalent compact disjunctive MILP using integer starts, binary assignments and binary pair-order variables. The serial horizon `H=max release+sum durations` supplies a justified big-M. The original time-indexed solver remains available for small reference tests. The corrected run `20261027T135208.655940Z` produced independently valid schedules; 7/14-job solves proved optimal, while all 24-job results are explicitly labelled time-limit feasible with their nonzero MIP gaps.

Lesson: an exact formulation is not automatically operationally useful; model size affects whether a solver finds even an incumbent. Preserve no-incumbent failures, use a more compact valid formulation, and never convert a time-limited feasible result into an optimality claim. Commit reference pending user Git work.

## Track G issue 05 — obsolete xv6 `copyout` signature (2026-10-27)

The first AI-generated scheduler-statistics implementation called `copyout(pagetable, address, source, length)`, matching an older xv6 interface. The pinned source requires `copyout(pagetable, process_size, address, source, length)`. GCC rejected the build with argument-conversion errors and “too few arguments”; no kernel was booted from the failed build.

Fix: pass `p->sz` as the second argument, following other copyout sites in the same pinned tree. Both experiment kernels then built, and the modified kernel later passed all 74 usertests. Lesson: inspect the exact pinned teaching-OS API rather than relying on patches or memory from another xv6 revision. Commit reference pending user Git work.

## Track G issue 06 — benchmark parser and console-marker races (2026-10-27)

The first Track G host parser used positional regex captures but called `groupdict()`, causing a post-guest `KeyError('workload')`. A second debug attempt treated the end of the currently received buffer as the end of a marker line. Because xv6 prints one character at a time, the harness sent the next command between characters of `BENCH_END workload=cpu`, corrupting the console. Concurrent children could likewise interleave result lines. Failed evidence is preserved in `results/G_stride/20261027T140729.093299Z` and `20261027T140946.775926Z`.

Fix: use named regex groups, require a newline-terminated completion marker, send fixed-size binary result structures through an xv6 pipe and let only the parent print sorted result lines. The debug rerun completed both policies, and the final 48-record run passed all 22 independent audit checks. Lesson: interactive console reads can end mid-line; use complete framing and single-writer output for machine-parsed guest evidence. Commit reference pending user Git work.

## Track J issue 07 — pytest root produced unusable node IDs (2026-10-28)

The first AI-generated coverage profiler changed into the vendored codebase but allowed pytest to discover the parent repository's `pyproject.toml`. Full profiling succeeded, but saved node IDs began with `vendor/track_j/more_itertools-10.7.0/`; selected-suite replay from inside that directory then failed with “file or directory not found.” Evidence is preserved in `results/track_j_test_selection/20261028T085216.965078Z`.

Fix: pass `--rootdir=.` consistently to profiling, benchmark and mutation pytest invocations. The corrected profiles use portable `tests/...` node IDs. Lesson: pytest node IDs depend on root discovery, so working-directory changes alone do not define a reproducible test identity. Commit reference pending user Git work.

## Track J issue 08 — timing-dependent coverage invalidated an ILP replay (2026-10-28)

In the corrected debug run `20261028T085303.958623Z`, the ILP selected `CallbackIterTests::test_abort` because the full profile observed it executing two queue-timeout lines. On replay, thread timing changed and those two lines were absent, even though the test passed. The runner correctly rejected the suite instead of claiming coverage preservation.

Fix: independently profile every selected suite and add a deterministic repair phase that can include the cheapest alternate test covering a replay gap. The selected final run needed no repair, and all four suites independently covered 2,030/2,030 target lines. Lesson: a coverage matrix is measured behavior, not an immutable test property; concurrency can make a mathematically feasible selection operationally incomplete. Commit reference pending user Git work.

## Track J issue 09 — interactive plotting backend ended the runner early (2026-10-28)

After the first corrected debug run had written selections, profiles, benchmarks and mutation results, importing the default macOS plotting backend terminated the process before the plot, summary and success pointer were written. The absence of `summary.json` and `latest-success.json` in `20261028T085539.363406Z` exposed the incomplete run.

Fix: set `MPLBACKEND=Agg` before importing pyplot and keep `MPLCONFIGDIR` inside the writable project. The final runner saved the figure and all terminal artifacts, and the verifier checked the plot exists. Lesson: benchmark automation must force a noninteractive plotting backend rather than inheriting GUI behavior. Commit reference pending user Git work.

## Track D issue 10 — manager-backed deques required forbidden IPC sockets (2026-10-28)

The first AI-generated work-stealing implementation used `multiprocessing.Manager().list()` for per-worker deques. Its unit test failed before executing work because the manager server tried to bind a local IPC socket and the managed environment returned `PermissionError: [Errno 1] Operation not permitted`. A manager would also add an unnecessary proxy process and serialization overhead.

Fix: replace manager proxies with fixed shared-memory arrays of task indices, shared left/right indices and one process-safe lock per deque. Owners pop from the right; thieves take from the left of another worker's deque. The corrected unit tests passed, the final run observed nine real steals at four workers, and the independent verifier reran every scheduler and passed 31 checks. Lesson: choose the smallest IPC mechanism that implements the needed semantics, and test process primitives in the actual execution environment. Commit reference pending user Git work.

## Track I issue 11 — queued Hamiltonian arithmetic corrupted annealing state (2026-10-29)

The first AI-generated quantum-annealing circuit constructed Hamiltonian observables and their interpolated arithmetic inside a QNode without stopping PennyLane's recording queue. Terms intended only as arguments to `ApproxTimeEvolution` were also recorded as circuit operations. The returned state was not normalized: probabilities summed to approximately 1e-90 instead of one, making the apparent annealing result physically invalid.

Fix: construct both the base Hamiltonians and every interpolated `(1-s) H_driver + s H_problem` expression inside `qml.QueuingManager.stop_recording()`, then queue only the intended time-evolution operation. Unit tests and the independent verifier now require probability normalization. In the selected run, each annealing distribution sums to one within 1e-13 and ground-state success increases from 0.2890 at T=0.5 to 0.9946 at T=8. Lesson: quantum-program objects can carry recording side effects; mathematical operator construction inside a circuit must be separated from executable operations, and every state/probability result needs a normalization invariant. Commit reference pending user Git work.

A separate plot-only bug compared Shor's in-memory factor tuple with a list and displayed a zero bar despite correct saved factors. Visual inspection caught it, the comparison was normalized with `list(...)`, and the final figure was regenerated. This is documented here as part of verification but is not counted as another independent failure case.
