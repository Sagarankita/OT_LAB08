# Track E — project-workflow job scheduling

## Process and evidence of an optimization opportunity

This main case study models a small project workflow: non-preemptive jobs have integer durations, release times, precedence dependencies and must be assigned to one of three identical workers. The process consumes elapsed project time. A fixed-order ready-list scheduler is a realistic baseline because it is fast and easy to operate, but it ignores which ready task unlocks the most downstream work.

The experiment uses 15 deterministic synthetic DAG instances: five seeds at 7, 14 and 24 jobs. Durations are 1–9 time units, releases are 0–6, and each job has at most three earlier predecessors. Identical instances are given to all methods. The 14-job baseline had mean makespan 34.4 while an exact ILP achieved 32.6; the 24-job baseline had mean 45.6 while critical-path priority achieved 43.0. This measured idle/ordering loss establishes an optimization opportunity, while the small instances also show that a heuristic is not always better.

## Mathematical formulation

For job `j` and worker `w`:

- `s_j` is a non-negative integer start time.
- `a_jw` is 1 exactly when job `j` is assigned to worker `w`.
- `o_ijw` orders jobs `i<j` if both use worker `w`.
- `C_max` is the makespan.

Minimize `C_max`, subject to:

```text
s_j >= release_j
sum_w a_jw = 1
s_j >= s_i + duration_i                         for every precedence i -> j
C_max >= s_j + duration_j                       for every job j
```

For every pair `i<j` and worker `w`, two disjunctive big-M inequalities enforce either `i` before `j` or `j` before `i` when both assignments are active. The horizon and `M` are both

```text
H = max_j(release_j) + sum_j(duration_j).
```

This is a valid serial-schedule upper bound. Since `s_j <= H-duration_j`, `M=H` is also sufficient to deactivate either non-overlap inequality. Time is discretized to integers because generated release times and durations are integers; this model would not represent arbitrary fractional start times without rescaling.

The objective is subject to hard release, precedence, assignment and non-overlap constraints. A separately recomputed lower bound is the maximum of total work divided by worker count and every `release_j + downstream_critical_path_length_j`.

## Compared methods

| Method | Variables / information | Constraints handled | Scale and guarantee | Hardware / parallelism |
|---|---|---|---|---|
| Fixed-order list baseline | Discrete ready job and worker choice | Constructively feasible | Linearithmic-scale heuristic; no quality guarantee | One CPU process; produces a parallel three-worker schedule |
| Critical-path list heuristic | Same, plus downstream path lengths | Constructively feasible | Fast heuristic; may improve or worsen baseline | One CPU process; schedule itself uses three workers |
| Disjunctive MILP | Integer starts, binary assignment/order, continuous makespan | Explicit linear constraints | Exact when status is optimal; otherwise bounded feasible incumbent | SciPy/HiGHS on local CPU; one-second solver limit |
| Exhaustive oracle | Topological dispatch and worker choices | Constructively feasible | Exact only for the five 7-job instances | One CPU process; excluded from measured method time |

Gradient methods do not naturally apply because assignments and order decisions are discrete and feasibility is combinatorial. A continuous relaxation would need rounding and repair. Quantum optimization could encode a discretized formulation as QUBO, but the many assignment/order/time variables, penalty selection and simulator or embedding overhead offer no automatic advantage at these sizes. The classical methods are directly verifiable and match the available hardware.

## Experimental protocol

Selected evidence: `results/track_e_scheduling/20261027T135208.655940Z`.

- Five fixed seeds: 71, 82, 93, 104 and 115.
- Three sizes: 7, 14 and 24 jobs; three identical workers.
- One unmeasured warm-up and three measured repetitions per method/instance.
- 135 raw records total: 5 seeds × 3 sizes × 3 repetitions × 3 methods.
- `time.perf_counter` measures method execution. Generation, exhaustive oracles and warm-ups are excluded; MILP timing includes model construction.
- Each saved schedule is independently checked for unique jobs, releases, durations, precedence, worker overlap and reported makespan.
- Small optima are independently re-enumerated rather than trusted from saved output.

## Results

Means include all seeds and repetitions. Makespan is in integer time units.

| Jobs | Method | Mean makespan | Mean runtime | Mean change vs baseline | Solver result |
|---:|---|---:|---:|---:|---|
| 7 | Fixed order | 24.4 | 0.0297 ms | 0% | Heuristic |
| 7 | Critical path | 24.8 | 0.0272 ms | −2.50% | Heuristic |
| 7 | MILP | 24.4 | 3.792 ms | 0% | 15/15 optimal |
| 14 | Fixed order | 34.4 | 0.0600 ms | 0% | Heuristic |
| 14 | Critical path | 33.0 | 0.0587 ms | +4.12% | Heuristic |
| 14 | MILP | 32.6 | 34.948 ms | +5.29% | 15/15 optimal |
| 24 | Fixed order | 45.6 | 0.1204 ms | 0% | Heuristic |
| 24 | Critical path | 43.0 | 0.1198 ms | +5.69% | Heuristic |
| 24 | MILP | 43.2 | 1004.319 ms | +5.33% | 15/15 time-limit feasible |

The 24-job MILP incumbents are **not claimed optimal**. All hit the one-second limit; their reported MIP gaps ranged from 4.65% to 25.49%, with mean 14.87%. The fast critical-path heuristic slightly beat their mean incumbent under that limit. Conversely, critical-path priority was worse than fixed order on one small seeded instance (three repeated records), demonstrating that its priority rule is not a dominance guarantee.

At 14 jobs, MILP removed the measured baseline gap but cost roughly 582 times the baseline runtime. At 24 jobs it was roughly four orders of magnitude slower than the heuristics and still did not prove optimality. Method selection therefore depends on whether solution quality, proof, or response time matters most.

## Failure and correction

The first AI-generated final configuration used the time-indexed reference MILP with a one-second limit. HiGHS reached the limit without a feasible incumbent on a 24-job case, so the run correctly failed at `results/track_e_scheduling/20261027T135049.886462Z`. Raising the limit to five seconds still produced no incumbent. The measured implementation was changed to the compact disjunctive formulation above; the time-indexed implementation remains a small reference. The corrected final run preserves time-limit statuses, dual bounds and gaps.

## Correctness and limitations

All 17 independent audit checks passed. The audit reconstructs schedules without calling the production validator, recomputes lower bounds and gaps, independently enumerates all five small optima, checks paired comparisons, verifies MILP statuses and hashes declared artifacts. The complete project suite has 38 passing tests.

Limitations: the data are synthetic; workers are identical; durations are deterministic; tasks are non-preemptive; communication, skills, calendars and reassignment costs are absent. Repetitions measure runtime variability but reuse deterministic instances. A deployment should use historical duration distributions, worker skills and robust or stochastic scheduling, then validate on held-out workflows.

```sh
.venv/bin/python -m pytest
.venv/bin/python scripts/run_track_e.py --config configs/track_e_debug.json
.venv/bin/python scripts/run_track_e.py --config configs/track_e_final.json
.venv/bin/python scripts/verify_track_e.py results/track_e_scheduling/20261027T135208.655940Z
```
