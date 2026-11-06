# Assignment 8 — Optimize Anything

## Execution plan: 2–3 days, one step at a time

**Current state: Steps 00–14 complete. All Tracks A–J have verified experiments, every selected verifier passes, and the detailed A–J technical report with all ten final answers is available as [Markdown](docs/REPORT.md) and a [submission-ready PDF](output/pdf/Assignment_8_Optimization_Technical_Report.pdf). The user explicitly declined Prompt 15; Git/publication and the genuine second-AI-tool comparison remain outstanding. See [docs/STATUS.md](docs/STATUS.md).**

This repository will contain ten focused optimization experiments (Tracks A–J), a technical report, reproducible results, and documented AI assistance. The target is three intensive working days; two days is a stretch target. Installation problems, xv6 debugging, and missing course material may require more time. Completion is determined by evidence, not the schedule.

### How to use this plan

All copyable execution prompts are in [Prompt.md](Prompt.md). Start with Prompt 00 and proceed sequentially.

1. Work through steps 00–15 in order. Each step is designed for a separate working session.
2. Copy its **continuation prompt** into the AI assistant. The assistant should read this README and existing work before editing anything.
3. Finish the step's checks before marking it complete. If a dependency blocks progress, document it and continue only with independent steps.
4. At the end of every session, update the progress table and `docs/STATUS.md` with files changed, commands run, checks passed, blockers, and the exact next action.
5. Save actual prompts, relevant responses, verification notes, and commit references as work happens. Do not reconstruct fictional evidence afterward.

Use this common instruction with every continuation prompt:

> Read README.md, docs/STATUS.md if present, and applicable AGENTS.md files. Execute only the requested step and its necessary prerequisites. Inspect existing work first. Implement, run, and verify the experiment; do not merely describe it. Preserve raw results and accurate AI logs. Do not invent measurements, failures, tool comparisons, or completed checks. Update the progress records and report the next step. Do not publish externally unless authorized.

## Scope and chosen problems

The main case study (Track E) is **resource-constrained job scheduling** using explicitly synthetic data. Schedule non-preemptive jobs on a fixed number of workers, subject to release times and precedence constraints. Minimize makespan. Start with small DAGs and identical workers; add other constraints only after this works.

This differs from Track D: E studies constrained schedule quality and a formal optimization model; D measures actual parallel execution, communication, and learning behavior. Do not submit the same experiment twice under different labels.

| Track | Bounded experiment | Baseline and alternatives |
|---|---|---|
| A | Small regression/ML tasks, quadratic and non-convex objectives, nonlinear least squares | GD, SGD, mini-batch SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, Newton, Gauss–Newton; curvature and convergence demonstrations |
| B | 0/1 knapsack and a constrained binary selection QUBO | Greedy, DP, ILP, explicit Branch-and-Bound, exhaustive verification; QUBO/Ising conversion |
| C | Track B QUBO with QAOA; tiny VQE and trainable circuits | Exact classical solution; parameter-shift, SPSA, quantum natural gradient; shots/noise/gradient scaling |
| D | Coverage selection, parallel task execution, distributed and multi-task learning | Exact versus greedy coverage; serial versus parallel schedules; centralized versus FedAvg; equal weighting versus MGDA/GradNorm |
| E | Synthetic job scheduling with release times and dependencies | List-order baseline, priority-based greedy, small-instance ILP |
| F | Toy compiler with lexer, parser, IR, executable code generation | Unoptimized versus four optimization passes; a formally optimized compiler decision |
| G | xv6 scheduling change, with a second feature as a full-credit target | Unmodified xv6 versus modified kernel, identical QEMU workloads |
| H | SQLite workload and budgeted index selection | Default/no added indexes, greedy selection, ILP or DP selection |
| I | Course quantum algorithms and quantum/classical ML comparison | Classical algorithm/model, query/gate counts, wall time, noise-model execution |
| J | Coverage-constrained test selection on a real open-source codebase | Full suite and naive subset versus greedy and ILP selection |

## Ground rules

- The exact Chapter 6 material is not yet available. Use the assignment's list provisionally and reconcile against class notes before declaring full coverage. Explicitly track unconfirmed syllabus items.
- All ten tracks are compulsory. A simulation of an OS scheduler does not satisfy Track G: modify actual xv6, boot under QEMU, run new user benchmarks and `usertests`, and submit the diff.
- A plain classical program described as a quantum circuit does not satisfy quantum execution. Use a quantum simulator and preserve circuit/configuration evidence.
- Do not claim quantum advantage from classical simulator timing. Separate theoretical query advantage, simulator measurements, and hardware/noise-model results.
- Negative results are acceptable when correct. Never tune the comparison just to force an improvement.
- Keep large ML training, cloud services, GUIs, elaborate applications, and unnecessary datasets outside scope.
- No optional extension should delay a compulsory requirement. A second xv6 feature is a full-credit target after the first feature is validated.

## Proposed repository layout

```text
README.md
requirements.txt                 # validated direct dependencies
requirements-lock.txt            # captured reproducible environment
tracks/
  A_continuous/ ... J_software/   # code, configs and track-specific instructions
tests/                          # independent correctness checks
configs/                        # seeds, sizes, tolerances, budgets
data/                           # generators and small public/synthetic inputs
results/<track>/<run-id>/        # raw CSV/JSON, logs, plans and metadata
figures/                        # generated from results
prompts/<A-J>/                  # original prompts, excerpts, revisions, commit links
prompts/setup/                  # planning and shared infrastructure AI use
docs/
  STATUS.md
  COVERAGE.md
  AI_TOOLS.md
  AI_FAILURES.md
  AI_TOOL_COMPARISON.md
  ENVIRONMENT.md
  REPORT.md
  DEMO.md
scripts/                        # reproducible run and report commands
vendor/                         # pinned external sources or acquisition instructions
patches/                        # xv6 modifications against recorded upstream commit
```

Create these as needed. Do not fill result files with sample numbers or mark empty directories as implemented experiments.

## Schedule and progress

Day 1: steps 00–05. Day 2: steps 06–10. Day 3: steps 11–15 and any unfinished verification. Start the second AI-tool comparison on Day 1 so access does not become a last-minute blocker.

| Step | Task | State |
|---|---|---|
| 00 | Requirements, provenance, and working environment | DONE (setup; external gaps documented) |
| 01 | xv6 baseline and toolchain smoke test | DONE (74 full usertests; independent audit passed) |
| 02 | Shared benchmark and verification infrastructure | DONE (12 tests; 50-record final smoke audit passed) |
| 03 | Track B: discrete optimization and shared QUBO | DONE (100 records; 19-check independent audit passed) |
| 04 | Track A: continuous optimization | DONE (50 records; 28-check independent audit passed) |
| 05 | Track E: main scheduling case study | DONE (135 records; 17-check independent audit passed) |
| 06 | Track F: compiler | DONE (270 records; 17-check independent audit passed) |
| 07 | Track G: xv6 modification and validation | DONE (minimum one-feature scope; 48 records, 74 usertests, 22-check audit) |
| 08 | Track H: database optimization | DONE (168 records; 20-check independent audit passed) |
| 09 | Track J: software-engineering optimization | DONE (671→242 tests; 22-check audit passed) |
| 10 | Track D: submodular and parallel experiments | DONE (1/2/4 workers; 31-check audit passed) |
| 11 | Track D: federated and multi-task experiments | DONE (45 records; 30-check audit passed) |
| 12 | Track C: quantum optimization | DONE (ideal/noisy simulators; 34-check audit passed) |
| 13 | Track I: quantum algorithms and QML | DONE (local ideal/noisy simulators; 41-check audit passed) |
| 14 | Coverage audit, reproducibility, and technical report | DONE (68 tests; all selected verifiers passed; report written) |
| 15 | Demonstration and public repository handoff | NOT REQUESTED (user declined; publication remains user-managed) |

### Step 00 — Requirements, provenance, and environment

**Actions:** Inspect available Python, Git, compilers, QEMU, RISC-V toolchain, storage and CPU resources. Select one supported Python environment and a small dependency set: NumPy/SciPy, plotting, pytest, an ILP interface, PyTorch, and one quantum framework. Verify current official installation instructions when installing. Prefer CPU-only experiments. Record versions and actual commands; do not assume this machine has working dependencies.

Create the status record, environment notes, AI register, prompt-log format and a provisional Chapter 6 coverage matrix. Preserve this planning interaction as AI assistance. Initialize Git if needed, with AI-assisted commit labels such as `[AI:Codex]`; distinguish subsequent manual work. Never guess the model version if unavailable—record it as unavailable.

**Done when:** Environment smoke tests run, installation blockers are recorded, every assignment requirement has a planned experiment, and AI provenance is being captured. Request class notes without blocking independent implementation.

**Continuation prompt:** `Execute step 00 of README.md: establish requirements, environment, Git provenance and the provisional coverage matrix. Check actual tools before selecting dependencies.`

### Step 01 — Prove xv6 can run early

**Actions:** Obtain xv6-riscv from its official source, record upstream commit and license, build and boot it under QEMU. Record the toolchain, QEMU version, CPU count and invocation. Run baseline `usertests` and save output. Select a scheduling modification compatible with that source version; avoid carrying a patch from an unrelated xv6 version.

**Done when:** xv6 boots and baseline tests pass, or a concrete environment blocker is recorded. Do not defer a broken toolchain until the report stage. If blocked, continue steps 02–06 while fixing access/toolchain separately; Track G remains incomplete.

**Continuation prompt:** `Execute step 01: establish and document a reproducible unmodified xv6 baseline under QEMU. Save boot and usertests evidence.`

### Step 02 — Shared experiment infrastructure

**Actions:** Implement configuration, seed handling, metadata capture, raw output storage and plot generation. Record hardware/OS, dependency versions, Git revision and working-tree changes, input hashes, solver settings and measurement units. Separate setup/compilation from execution timing, and report both where relevant.

Use small debug runs first, then an explicitly configured final run. Target at least five seeds for stochastic comparisons where feasible; report individual runs and mean/std or median/IQR. Use repeated timings for short deterministic tasks. State justified exceptions and timeouts. Never compare a tiny baseline instance with a different optimized instance.

**Done when:** A smoke experiment saves and reloads valid raw results; plots are generated from these results; invalid input and failed runs are represented honestly.

**Continuation prompt:** `Execute step 02: create a lightweight shared benchmark runner and independent validation conventions, then verify a smoke run.`

### Step 03 — Track B: exact and heuristic discrete methods

**Actions:** Generate seeded knapsack cases. Compare value/weight greedy, DP, ILP and an explicit Branch-and-Bound implementation. Use exhaustive enumeration on tiny instances to validate feasibility and optimal values.

Create a small constrained binary selection problem with a justified penalty, for example a fixed-cardinality selection objective. Derive its QUBO, spin conversion and constant offset. Verify every bitstring on tiny instances, show feasible optimal solutions at a sufficient penalty, and demonstrate how an insufficient penalty can fail. Export the identical coefficients and instances for Track C. Include feature selection if class notes require it separately.

**Done when:** Independent checks pass; exact/heuristic gaps and runtimes are saved; QUBO and Ising energies agree including offsets; the shared quantum input is frozen.

**Continuation prompt:** `Execute step 03 / Track B, including independently verified knapsack solvers and a constrained QUBO/Ising instance for Track C.`

### Step 04 — Track A: continuous methods and foundations

**Actions:** Implement or transparently use libraries for GD, SGD, mini-batch SGD, Momentum, NAG, AdaGrad, RMSProp and Adam on suitable tasks. Use Newton on a smooth small objective and Gauss–Newton on nonlinear least squares; they need not share an unsuitable benchmark.

Demonstrate parameters/loss/regularization, continuous versus discrete variables, convex/strongly convex/non-convex functions, a saddle and local/global minima. Show backpropagation separately from the optimizer, with finite-difference gradient checks. Include Hessian-vector products, a diagonal curvature approximation and Fisher-based curvature; add low-rank treatment if taught. Examine step sizes, noisy gradients/objectives and empirical convergence. State assumptions before mentioning theoretical rates.

**Done when:** Known-optimum and gradient checks pass, all named methods have real runs, and convergence/quality/runtime/sensitivity plots are generated. Use comparable gradient-evaluation or time budgets and explain second-order costs.

**Continuation prompt:** `Execute step 04 / Track A. Cover continuous methods, backpropagation, curvature and optimization foundations with small verified experiments and fair budgets.`

### Step 05 — Track E: main case study

**Formulation:** For job j, choose start time s_j and worker assignment. Minimize C_max. Enforce s_j >= release_j, predecessor completion before successor start, at most one active job per worker, and C_max >= s_j + duration_j. Explain any discretization, horizon, and big-M bounds in the ILP.

**Actions:** Measure a fixed-order list scheduler. Compare a critical-path or longest-ready-job heuristic and an ILP on identical DAG instances. Use small instances for optimality verification and larger ones for scaling with solver time limits. A timed-out feasible solution is not a proven optimum: preserve status, bound and gap.

**Done when:** An independent validator checks every schedule. Record makespan, runtime, feasibility and lower-bound/optimality gaps. Include a strategy table covering variable types, constraints, size, exact/heuristic choice, hardware and parallelism. Explain why quantum/gradient methods are not automatically appropriate here.

**Continuation prompt:** `Execute step 05 / Track E: formulate, implement and benchmark synthetic resource-constrained job scheduling, with an independent schedule validator.`

### Step 06 — Track F: compiler

**Actions:** Build a small toy language with lexer, parser, IR and code generator targeting executable bytecode or another documented target. Implement constant folding, constant propagation, dead-code elimination and common-subexpression elimination with explicit semantics. Specify integer arithmetic and side effects.

Formulate a genuine compiler decision, such as choosing a pass sequence under a compile-time budget; solve it with a named greedy or DP method and compare with a fixed order. Keep tuning programs separate from evaluation programs. Do not label the compiler's ordinary implementation as the required optimization formulation.

**Done when:** Source execution/reference interpretation, unoptimized generated code and optimized code agree on known and generated programs. Include reassignment/side-effect edge cases. Save static/dynamic instruction counts, executable size, execution time and compile time with units.

**Continuation prompt:** `Execute step 06 / Track F: build the compiler pipeline, four correct passes, and a Chapter 6 formulation of a compiler decision. Verify output equivalence and measure costs.`

### Step 07 — Track G: actual xv6 optimization

**Actions:** Implement one bounded kernel feature, preferably a scheduling policy with well-defined accounting. Add only the user-visible controls/statistics necessary for its experiment. Document locking, process-state transitions and fairness implications. Run tests after every kernel change.

Create at least three user workloads: CPU-bound, mixed CPU/sleep, and latency-sensitive/background competition. Compare baseline and modified kernels with the same QEMU configuration and workload parameters; use one emulated CPU for the first scheduling comparison. Record scheduling metrics with their clock resolution. Add a second independent feature only after the first passes; document whether it meets the full-credit target.

**Done when:** Modified xv6 boots, passes `usertests`, preserves workload outputs and produces before/after measurements for all three workloads. Submit a reproducible diff against the pinned upstream revision, benchmark sources and full logs. Explain limitations of QEMU timings.

**Continuation prompt:** `Execute step 07 / Track G on the validated xv6 baseline. Implement one kernel optimization, test after changes, benchmark three workloads, and save a reproducible patch.`

### Step 08 — Track H: database

**Formulation:** Choose binary index variables x_i, maximize measured workload benefit or minimize latency, subject to measured index storage <= budget. Explain that additive benefit estimates can miss interactions and validate the selected set by executing the complete workload.

**Actions:** Generate a documented relational SQLite dataset, starting small and scaling to a useful size (for example 100,000 rows in the main table). Create a multi-query workload and several candidate indexes. Measure the default/no-added-index baseline, greedy selection and ILP or DP selection. Include an all-index reference if useful.

**Done when:** Preserve `EXPLAIN QUERY PLAN` output, repeated query latencies, workload throughput and storage for each selected set. Compare results with an ordering-aware or multiset-aware checker. Document warm-cache conditions and index-build costs; do not call SQLite plans `EXPLAIN ANALYZE` results.

**Continuation prompt:** `Execute step 08 / Track H: benchmark budgeted SQLite index selection using greedy and an exact method against the default optimizer, with result equivalence checks.`

### Step 09 — Track J: realistic software-engineering task

**Actions:** Select a public, licensed, pinned codebase with at least 1,000 meaningful source lines, a runnable test suite and usable coverage. Record actual source size, license and provenance; do not inflate line count. Profile the full suite and gather per-test coverage and durations.

**Formulation:** Choose tests to minimize total measured runtime while preserving the target coverage set. Compare full-suite, deterministic naive coverage-preserving selection, greedy set cover and ILP. All reduced-suite comparisons must use the same coverage requirement. Separate selection/profiling overhead from execution savings.

**Done when:** Run selected suites, recompute retained coverage, and report runtime/selection overhead, test counts and maintainability implications. Coverage preservation does not establish equal fault detection: use a small seeded-mutation check if feasible, otherwise state that limitation. Document why repeated CI runs might amortize profiling cost.

**Continuation prompt:** `Execute step 09 / Track J: select a suitable real codebase, profile its tests, formulate test selection, compare naive/greedy/ILP and verify retained coverage.`

### Step 10 — Track D: submodular and parallel experiments

**Actions:** Use maximum coverage under a cardinality budget. Check normalization, non-negativity, monotonicity and diminishing returns by exhaustive small-set tests. Compare greedy with exhaustive optimum on small cases. State the conditions behind the 1-1/e guarantee and do not transfer it to unrelated objectives.

For actual parallel execution, compare serial, static balanced assignment and dynamic task distribution/work stealing. A centralized shared queue is dynamic scheduling, not work stealing; implement per-worker deques and actual stealing if claiming the latter. Include uneven task lengths, locality and synchronization/communication costs. Demonstrate data, model and pipeline parallelism with small real process-based computations; measure overhead even if they are slower.

**Done when:** Outputs agree, multiple worker counts are tested within available cores, and speedup S_p=T_1/T_p and efficiency S_p/p are reported. Use fixed-size and scaled-size experiments to discuss Amdahl and Gustafson respectively. Label any assumptions or fitted serial fractions.

**Continuation prompt:** `Execute step 10 / Track D: verify submodular greedy assumptions and measure real parallel scheduling plus data/model/pipeline parallelism, including overhead and correctness.`

### Step 11 — Track D: distributed and multi-task optimization

**Actions:** Compare centralized learning and FedAvg on fixed synthetic or documented public data. Vary local updates and communication rounds, include an IID/non-IID split, weight client updates by sample count, and report both local compute and transmitted parameter bytes. Distinguish same-total-update comparisons from communication-budget comparisons.

Train a small shared model with two task heads. Compare equal weighting, MGDA and GradNorm with the same split, initialization seeds and training budget. Validate MGDA's constrained weighting on a known two-gradient case. Ensure GradNorm's learned weights receive the intended gradient and normalization; do not substitute a heuristic while retaining the algorithm's name.

**Done when:** Save per-task metrics, communication rounds/bytes, losses and runtime; include implementation checks and explain task trade-offs. Explicitly label local simulation rather than geographically distributed hardware.

**Continuation prompt:** `Execute step 11 / Track D: implement verified FedAvg, MGDA and GradNorm comparisons with explicit compute and communication budgets.`

### Step 12 — Track C: quantum optimization

**Actions:** Load Track B's exact QUBO and construct its cost Hamiltonian with tested basis/sign conventions. Run QAOA and compare expected objective, feasible-sample rate and best sampled feasible solution with the classical optimum. Run small VQE against exact diagonalization and a small trainable PQC/QNN. Demonstrate parameter-shift, SPSA and metric-based quantum natural gradient.

Use analytic versus finite-shot measurements, several shot budgets, and a documented hardware noise model where supported. Measure gradient variance over seeds as qubits/depth increase. A small decreasing-variance experiment is evidence of a trend, not proof of a general barren-plateau law. Account for the shot/circuit cost of gradients and metric estimation.

**Done when:** Circuit and derivative checks pass; raw seeds, shots, circuit calls and optimizer budgets are saved; noisy results are labelled simulated. Include basic quantum concepts and the role of optimization in QML.

**Continuation prompt:** `Execute step 12 / Track C: run QAOA on Track B's verified QUBO, VQE and PQC optimization, including gradient methods, shot costs, noise and gradient-variance experiments.`

### Step 13 — Track I: quantum algorithms and QML

**Actions:** Provisionally cover Deutsch–Jozsa, Bernstein–Vazirani, Grover, QFT, QPE, a small Shor demonstration and quantum annealing; reconcile this list with actual class material. Write circuits for at least two algorithms directly and test known outputs/distributions. If Shor is compiled for a fixed instance, disclose the simplification and do not present it as general factoring. Classical simulated annealing is a classical baseline, not a quantum-annealing implementation; use a valid small quantum annealing evolution/model and discuss embedding limits.

Compare Grover and linear search using matched oracle assumptions, query counts, gate counts and simulator wall time. Train a variational classifier/QNN and a classical model on the same train/validation/test data; also demonstrate a quantum kernel if taught. Match tuning effort and prevent data leakage. Include a noisy execution and reuse the documented noise model from C if appropriate, linking distinct outputs.

**Done when:** Save accuracy, training runtime, parameter count and circuit evaluations, algorithm verification evidence and all taught-method coverage. Discuss state preparation/data loading, simulation overhead, noise, embedding connectivity and chain overhead for annealers, and compare these limits with actual parallel results from D. Real hardware is optional when access is unavailable; simulated noise must be explicit.

**Continuation prompt:** `Execute step 13 / Track I: verify the required quantum algorithms and a fair QML/classical comparison. Clearly disclose circuit simplifications, simulation and hardware noise models.`

### Step 14 — Audit, reproduce and write the report

**Actions:** Reconcile COVERAGE.md with the exact syllabus if obtained. Mark each item demonstrated, missing or unconfirmed; a description alone is not coverage. Run a clean-install smoke workflow and the documented final commands. Check that figures and claims trace to raw results and that each track has a correctness check.

Write `docs/REPORT.md` with an A–J section. Each section must include process, mathematical formulation, baseline, method rationale, setup, correctness evidence, results with units/variability, limitations and code/result links. Answer all ten final assignment questions. Include honest unsuccessful optimizations and why they failed. Export to PDF only after content is verified if a PDF is desired/required.

**Done when:** All mandatory items have evidence or are explicitly disclosed as incomplete; report tables match raw files; setup commands work; licenses and AI disclosures are complete. No unchecked step may silently become “done.”

**Continuation prompt:** `Execute step 14: audit every assignment requirement against real artifacts, rerun reproducibility checks and write the evidence-based technical report. List unresolved gaps explicitly.`

### Step 15 — Demonstration and repository handoff

**Actions:** Prepare a short demonstration script showing formulation, one before/after comparison, independent correctness checks, xv6 boot/benchmark evidence, quantum limitations and AI corrections. Include viva questions and small code modifications the student can practice without AI.

Review files for secrets, private data, unnecessary binaries and license compliance. Ensure the public README includes setup, execution, results navigation, AI tools table and external-source acknowledgments. Verify the remote and obtain explicit publication authorization before creating/pushing a public repository if not already authorized.

**Done when:** Public repository URL is verified, clean-clone instructions are tested, report is accessible, and the student can explain the code. Never fabricate a publication URL or mark publication complete after a local commit alone.

**Continuation prompt:** `Execute step 15: prepare the demonstration, viva practice and submission audit. Prepare a concrete publication-ready repository; publish only with authorization and verify the resulting URL.`

## Mandatory AI evidence, maintained throughout

For each substantial implementation/revision, save: date, tool and actual model identifier if available, track, original prompt, relevant response excerpt, revised prompt if any, changed files, commit reference and independent verification note. Preserve available genuine chat exports/screenshots; redact secrets before publication.

The final README must contain the AI tools register itself, not only a link. Initial register:

| Tool | Model | Tracks/scope | Date | Work |
|---|---|---|---|---|
| Codex | Exact identifier not recorded in this planning artifact | Shared planning for A–J | 2026-10-26 | Experiment selection, execution plan and README drafting |
| Codex | Exact model identifier unavailable | Shared setup and G baseline | 2026-10-27 | Sequential prompts, environment, xv6 source acquisition, build/test harness and verification |
| Codex | Exact model identifier unavailable | Shared benchmark infrastructure | 2026-10-27 | Result schema, seed/timing metadata, failure handling, plots, tests and independent audit |
| Codex | Exact model identifier unavailable | Track B discrete optimization | 2026-10-27 | Greedy/DP/ILP/Branch-and-Bound, oracle, QUBO/Ising, tests and evidence |
| Codex | Exact model identifier unavailable | Track A continuous optimization | 2026-10-27 | Eight first-order methods, second order, backpropagation, curvature and noisy objectives |
| Codex | Exact model identifier unavailable | Track E scheduling case study | 2026-10-27 | DAG scheduling formulation, baseline, critical-path heuristic, MILPs, oracle and verification |
| Codex | Exact model identifier unavailable | Track F compiler optimization | 2026-10-27 | Lexer/parser/IR/bytecode, four passes, DP pass ordering and equivalence audit |
| Codex | Exact model identifier unavailable | Track G xv6 optimization | 2026-10-27–28 | Stride scheduler, control/statistics, three workloads, QEMU validation and patch |
| Codex | Exact model identifier unavailable | Track H database optimization | 2026-10-28 | SQLite data/workload, greedy/DP index selection, plans, storage and audit |
| Codex | Exact model identifier unavailable | Track J software-engineering optimization | 2026-10-28 | Pinned public codebase, coverage profiling, naive/greedy/ILP selection, mutations and audit |
| Codex | Exact model identifier unavailable | Track D submodular and parallel optimization | 2026-10-28 | Axiom/guarantee checks, real scheduling/work stealing, process parallelism and audit |
| Codex | Exact model identifier unavailable | Track D distributed and multi-task optimization | 2026-10-28 | Weighted FedAvg, communication accounting, MGDA, GradNorm and audit |
| Codex | Exact model identifier unavailable | Track C quantum optimization | 2026-10-28 | QAOA, VQE, PQC/QNN, parameter-shift, SPSA, QNG, shots/noise and audit |
| Codex | Exact model identifier unavailable | Track I quantum algorithms and QML | 2026-10-29 | Direct circuits, compiled Shor, unitary annealing, matched QML/classical models and audit |

Update this with actual implementation sessions. Do not represent planning as completed code generation.

**Second-tool comparison:** On Day 1, send the same bounded task to Codex and an actually accessible second AI tool—for example, a knapsack implementation plus edge-case tests. Preserve both outputs, prompt iterations, measured response timing if captured, correctness results and hallucinations. Obtain/export the second tool's actual interaction; do not let one assistant pretend to be another. If access is unavailable, flag this requirement as blocked.

**Failure log:** Record at least three real cases where AI output was wrong, unsafe, inefficient or misleading. Include original evidence, failing check, fix and lesson. Deliberately injected mutation tests can validate tests, but must not be misrepresented as genuine AI mistakes.

## Final submission checklist

- [x] All ten tracks have formulation, baseline, fair comparison and correctness evidence.
- [ ] Every taught Chapter 6 method maps to code and a real result, including foundations and strategy selection.
- [x] Main case study includes assumptions, mathematical formulation and method-selection table.
- [x] Compiler includes four passes, a formulated compiler decision and output equivalence evidence.
- [x] xv6 includes kernel changes, three workloads, boot/usertests evidence and reproducible patch; second-feature status is explicit.
- [x] DBMS includes multiple candidate indexes, two optimization methods, plans, storage and identical query results.
- [x] Quantum work includes algorithms, QML comparison, cost accounting and explicit simulation/noise labels.
- [x] Parallel/distributed work includes real measurements, FedAvg, MGDA and GradNorm.
- [x] Track J uses a non-trivial codebase with provenance, profiling and verified constraints.
- [x] Raw data, plots, settings, seeds, dependency versions and execution commands are preserved.
- [ ] README AI register, per-track prompt logs, real failure cases and real two-tool comparison are complete.
- [x] Report answers all ten final questions and links all A–J artifacts.
- [ ] Public GitHub repository, report and demonstration are ready and verified.

**Next action: add the genuine second-AI-tool evidence, then perform the user's Git commits and public GitHub publication. Prompt 15 was explicitly declined.**

## Local setup and user preferences (2026-10-27)

The user handles all Git work and publishing. Do not initialize, stage, commit or push; earlier Git instructions in this plan are superseded. Commit references remain pending until supplied by the user. This folder currently inherits a Desktop repository, which must not be used for staging project work.

Use Python 3.11 in the local environment:

```sh
source .venv/bin/activate
python scripts/check_environment.py
python -m pip check
```

To recreate the environment on a compatible machine:

```sh
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python scripts/check_environment.py
```

The lock captures this macOS arm64 environment, not a guarantee of cross-platform compatibility. ILP uses SciPy/HiGHS; quantum simulation uses PennyLane; training uses PyTorch on CPU. See [environment notes](docs/ENVIRONMENT.md) and [coverage checklist](docs/COVERAGE.md).

## xv6 baseline evidence (Prompt 01)

Unmodified pinned xv6 built and booted with QEMU 11.1.2 and RISC-V GCC 16.2.0. Full `usertests` passed all 74 cases with one emulated CPU and 128 MiB RAM. The successful test run took about 145.7 host seconds; this is test-runtime evidence, not an optimization speedup. All 82 original source files remain unchanged.

- [Track G instructions and proposed feature](tracks/G_xv6/README.md)
- [Successful baseline console log](results/G_baseline/20261027T132857.135570Z/console.log)
- [Build/run metadata](results/G_baseline/20261027T132857.135570Z/metadata.json)
- [Independent verification](results/G_baseline/20261027T132857.135570Z/verification.json)

Initial path-quoting, console-harness, syscall-API and marker-framing failures are preserved and disclosed in [the failure log](docs/AI_FAILURES.md). Track G's weighted-stride kernel, three workloads, full usertests logs and reproducible patch are complete; the optional second kernel feature is explicitly not implemented.
