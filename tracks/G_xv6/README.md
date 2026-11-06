# Track G — xv6 weighted-stride scheduling

## Source, scope and objective

Official xv6-riscv is pinned at commit `e93dabeb98431235c6e1a101cb96c73c923fdbc4`. The untouched source and its 82-file SHA-256 manifest remain in `vendor/xv6-riscv-baseline`. Two separate experiment trees contain an identical benchmark program:

- `vendor/xv6-riscv-track-g-baseline`: original process-table round-robin kernel;
- `vendor/xv6-riscv-track-g-stride`: weighted-stride kernel plus two control/statistics syscalls.

The optimized process is CPU scheduling. The objective is to allocate more scheduling opportunities to explicitly higher-weight processes and reduce their completion time, while preserving process-state correctness and all existing xv6 behavior. One kernel feature is implemented, satisfying the assignment minimum. A second independent kernel feature was **not implemented**, so the stated full-credit target remains incomplete.

## Scheduler design

Each process stores:

```text
weight in [1,100], default 10
stride = 10,000,000 / weight
pass = virtual runtime
dispatches = scheduler selections
run_ticks = timer interrupts while running
```

At each scheduling decision, the kernel snapshots all `RUNNABLE` processes and chooses the lowest `pass`. It then acquires only that candidate's lock, rechecks both state and pass to handle another CPU racing for the same process, sets it to `RUNNING`, adds its stride, increments dispatches and context-switches. The process later changes its state and returns to the scheduler under the existing xv6 lock protocol.

`setweight` validates 1–100 and changes the current process's weight/stride. `getschedstats` copies current weight, stride, pass, dispatch count and timer-tick count to user space. Children inherit their parent's weight and pass; starting every child at zero would reward fork-heavy programs unfairly. Timer accounting occurs immediately before timer-driven yield in both user and kernel trap paths.

The implementation rechecks a candidate under its lock and avoids holding multiple process locks across a context switch. The experiment measures one emulated CPU. The snapshot/recheck logic is intended to avoid selecting one process simultaneously on multiple CPUs, but the results are not evidence of multi-CPU scalability or fairness.

## Workloads and fair comparison

The exact same `user/schedbench.c` is compiled into both kernels. Child processes send binary result structures through a pipe; only the parent prints, preventing concurrent console output from corrupting evidence.

Three workloads run three times each:

1. **CPU-bound:** three equal-work jobs with advertised weights 1, 3 and 6.
2. **Mixed CPU/sleep:** a weight-1 CPU background job and weight-8 interactive job alternating short computation with five one-tick pauses.
3. **Latency/background competition:** two weight-1 background jobs and a later weight-10 target; response and completion ticks are recorded.

Both kernels use QEMU `virt`, 128 MiB RAM, one RISC-V CPU, the same compiler, timer and workload constants. Checksums must match for every policy/workload/role/repetition. The timer is programmed for about 0.1 seconds, so all guest timing has coarse one-tick resolution.

## Results

Selected evidence: `results/G_stride/20261027T141300.613368Z`.

| Workload metric | Round-robin mean | Stride mean | Change | Interpretation |
|---|---:|---:|---:|---|
| CPU job, weight 1 completion | 5 ticks | 6 ticks | 20% worse | Expected cost of lower share |
| CPU job, weight 3 completion | 3 ticks | 3 ticks | Tie | No measurable change |
| CPU job, weight 6 completion | 4 ticks | 3 ticks | **25% better** | Intended high-weight benefit |
| Mixed background completion | 2 ticks | 2 ticks | Tie | No measurable change |
| Mixed interactive completion | 5 ticks | 5 ticks | Tie | Five sleeps dominate elapsed time |
| Latency target response/completion | 2 ticks | 3 ticks | **50% worse** | Negative result |

All three repetitions produced the same tick values; this reflects deterministic QEMU conditions and coarse resolution, not broad statistical certainty. The fixed-work CPU result demonstrates a trade: the highest-weight job completes sooner while the lowest-weight job completes later. It does not prove long-run proportional shares because jobs finish at different times and recorded run ticks are small.

The latency regression is important. The child inherits the parent's current virtual time, and a child can only call `setweight` after its first dispatch. Therefore its new high weight cannot improve the initial response being measured. A production design could provide a parent-controlled `setweight(pid, weight)` interface and a carefully bounded arrival/wakeup pass rule, but those are not implemented or claimed here.

## Correctness evidence

- Both benchmark kernels built successfully.
- All 48 raw result records are paired and all computation checksums match.
- The stride records contain non-negative run ticks and positive dispatch counts.
- The modified kernel booted and started all 74 upstream tests in exact order.
- It printed `ALL TESTS PASSED`, returned to the shell and produced no panic.
- Full modified-kernel usertests took 149.20 host seconds, compared with 145.70 seconds for the earlier pristine baseline run. These are regression-suite timings, not a scheduler throughput benchmark.
- The independent Track G verifier passed 22/22 checks.
- `xv6-stride.patch` applies cleanly in dry-run mode to the pinned source.

## Failures found during AI-assisted development

The first kernel build used an older four-argument `copyout` call; this pinned xv6 requires the process-size argument. The compiler caught it and the call was corrected. Two early benchmark harnesses also failed: one expected named regex groups that were not defined, and another reacted to a partial `BENCH_END` line, injecting the next command into guest output. Results were moved through pipes and markers now require a newline. Failed runs remain in `results/G_stride` and are disclosed in `docs/AI_FAILURES.md`.

## Limitations

- Only one emulated CPU was measured; QEMU host time and guest ticks are not bare-metal performance.
- Tick resolution is about 0.1 seconds, limiting sensitivity.
- The 64-bit pass counter is not renormalized; overflow is unrealistic for this bounded experiment but remains a theoretical long-uptime limitation.
- Sleepers keep their previous pass, which can create either catch-up bursts or delayed service depending on the runnable set.
- Static user-selected weights can be abused and do not adapt to workload behavior.
- A second independent kernel feature was not implemented.

## Reproduction

```sh
# Fast benchmark smoke run without full usertests
.venv/bin/python scripts/run_track_g.py --cpus 1 --repetitions 1 --skip-usertests --timeout 300

# Final comparison and modified-kernel regression suite
.venv/bin/python scripts/run_track_g.py --cpus 1 --repetitions 3 --timeout 900

.venv/bin/python scripts/analyze_track_g.py results/G_stride/20261027T141300.613368Z
.venv/bin/python scripts/verify_track_g.py results/G_stride/20261027T141300.613368Z
```

The runner rebuilds both trees with a quoted deterministic-path flag because this workspace path contains a space. It boots QEMU directly, preserves complete console/build logs and terminates QEMU only after a recovered shell marker.
