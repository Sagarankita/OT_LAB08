# Track D — submodular and parallel optimization

This is Prompt 10's portion of Track D. Distributed FedAvg and multi-task MGDA/GradNorm remain Prompt 11 work.

## 1. Submodular maximum coverage

Given candidate sets `A_i`, choose at most `k=4` to maximize

```text
f(S) = | union of A_i for i in S |
```

The greedy method repeatedly chooses the largest marginal coverage gain. On a seven-set instance, exhaustive enumeration checked all 128 subsets and 5,103 valid diminishing-return comparisons:

- `f(empty)=0` (normalized);
- `f(S)>=0` (nonnegative);
- `A subset B => f(A)<=f(B)` (monotone);
- for `A subset B` and `e not in B`, `f(A+e)-f(A) >= f(B+e)-f(B)` (diminishing returns).

Ten seeded 12-set instances were compared with exhaustive `C(12,4)=495`-choice optima. Greedy's mean ratio was 0.9865 and minimum ratio 0.9375.

The `1-1/e` guarantee applies because this objective is normalized, nonnegative, monotone and submodular, the constraint is a pure cardinality budget, and greedy uses exact marginal gains. The observed ratios are not a stronger theoretical guarantee. This guarantee does not apply to runtime scheduling, data parallelism, model parallelism or pipeline parallelism.

## 2. Scheduling experiment

Thirty-two deterministic CPU tasks have deliberately uneven costs and four affinity blocks. All methods execute the same task function and return identical `(task_id, checksum)` outputs.

- Serial: one process, no startup communication.
- Static balanced: longest-processing-time assignment using known work estimates, then fixed per-worker buckets.
- Dynamic: workers pull from one centralized shared queue. This is explicitly not called work stealing.
- Work stealing: each worker owns a shared-memory deque with independently locked left/right indices; it pops locally from the right and steals from another worker's left.

Each process-based run includes spawn, serialization, synchronization and collection. `residual overhead = elapsed - maximum worker compute time` is reported as a combined overhead proxy, not a pure communication measurement. A task whose affinity block differs from its executing worker is counted as remote; this is a logical locality model, not measured NUMA traffic.

### Fixed-size results

One warm-up and three measured runs used 1, 2 and 4 workers on a 10-logical-CPU Mac. `S_p=T_1/T_p`; `E_p=S_p/p`.

| Method | Workers | Mean (s) | SD (s) | Speedup | Efficiency | Residual overhead (s) | Remote tasks | Steals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Serial | 1 | 0.820 | 0.001 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| Static | 2 | 0.679 | 0.009 | 1.207 | 0.604 | 0.250 | 16 | 0 |
| Static | 4 | 0.524 | 0.003 | 1.564 | 0.391 | 0.287 | 24 | 0 |
| Dynamic | 2 | 0.680 | 0.003 | 1.206 | 0.603 | 0.252 | 15.3 | 0 |
| Dynamic | 4 | 0.524 | 0.001 | **1.565** | 0.391 | 0.274 | 25.7 | 0 |
| Work stealing | 2 | 0.688 | 0.004 | 1.192 | 0.596 | 0.259 | 3.3 | 3.3 |
| Work stealing | 4 | 0.540 | 0.006 | 1.519 | 0.380 | 0.276 | 9 | 9 |

The one-worker process variants take about 1.078 seconds (0.76x) because they still pay spawn/IPC overhead. Work stealing preserved logical locality better than the other process schedules but was slightly slower because deque locks and stealing also cost time.

### Amdahl and Gustafson interpretation

For `p>1`, the fitted Amdahl serial fraction is

```text
s = (1/S_p - 1/p) / (1 - 1/p)
```

At four workers the fitted fractions are 0.519 static, 0.519 dynamic and 0.545 work stealing. These are empirical fits that absorb Python spawn, serialization and scheduling overhead; they are not intrinsic fractions of the algorithm. The one-worker process measurements are not fitted because Amdahl's inversion is undefined there.

For scaled work stealing, task count grows from 8 at one worker to 32 at four. At four workers the same-size serial reference took 0.828 seconds and work stealing took 0.537 seconds: measured scaled speedup 1.542 and efficiency 0.385. Gustafson's prediction using the fixed-size fitted fraction is 2.366, substantially higher. The gap shows that its constant-serial-fraction assumption does not model spawn and synchronization costs well here.

## 3. Data, model and pipeline parallelism

All demonstrations use real spawned processes and identical results, but their intentionally small workloads expose overhead limits.

| Experiment | Parallel configuration | Communicated bytes | Serial (s) | Parallel (s) | Speedup |
|---|---|---:|---:|---:|---:|
| Data: modular transform/reduction | 4 processes | 2,400,000 | 0.0167 | 0.2517 | 0.0662 |
| Model: feature/tensor-split linear prediction | 4 processes | 3,169,024 | 0.0000134 | 0.2398 | 0.0000559 |
| Pipeline: three queued stages, 12 batches | 3 stage processes | 2,304,000 | 0.00936 | 0.2777 | 0.0337 |

The tensor/model experiment splits features and weights across workers, sums partial predictions and checks `allclose`. The pipeline connects three persistent stage processes with bounded queues. These methods are slower because computation per message is tiny compared with process startup, serialization and queue transfer. This honest negative result defines when parallelism should not be selected; larger compute-to-communication ratios or persistent worker pools would be needed.

## Correctness and reproducibility

- Every fixed and scaled scheduler output matched the serial hash.
- Data reductions were equal, model predictions passed strict `allclose`, and pipeline batch checksums matched.
- Actual stealing occurred at 2 and 4 workers.
- Three measured repetitions and one warm-up were saved with worker records, output checksums, timing variability, residual overhead, affinity violations and communication bytes.
- The independent verifier reproduced exact maximum-coverage values and formulas, then ran fresh small process-based correctness checks; 31/31 checks passed.

Selected run: `results/track_d_parallel/20261028T092127.549032Z/`.

```sh
.venv/bin/python scripts/run_track_d_parallel.py --config configs/track_d_parallel_debug.json
.venv/bin/python scripts/run_track_d_parallel.py --config configs/track_d_parallel_final.json
.venv/bin/python scripts/verify_track_d_parallel.py results/track_d_parallel/20261028T092127.549032Z
.venv/bin/python -m pytest
```

Git and publication remain user-managed.
