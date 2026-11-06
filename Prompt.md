# Sequential project prompts — Assignment 8

Created: 2026-10-27. These are instructions for future execution, not evidence that any experiment has run.

## Current user instruction

The original execution period used local-only Git instructions. On 6 November 2026, the user explicitly superseded that instruction and authorized cleanup and publication to `https://github.com/Sagarankita/OT_LAB08`. Historical prompt logs preserve their original wording. Steps 00–14 have been executed. The user explicitly declined Prompt 15; see docs/STATUS.md for remaining submission gaps.

## How to proceed

- Run prompts **00 through 15 in order**, one prompt per working session. Each numbered prompt maps to the same README step.
- Copy the entire fenced block for the current prompt, or say: **“Read Prompt.md and execute Prompt 00 only.”** Replace the number as you progress.
- Wait for verification and the progress update before moving on. If a step is unfinished, resume it instead of skipping its checks. A recorded environment blocker may permit independent later work, but the blocked requirement remains incomplete.
- Run the separate second-AI-tool task below during Day 1, after setup. Its real outputs are required for the disclosure comparison.
- README.md is the experiment specification; docs/STATUS.md will be the execution handoff. If implementation changes the plan, update both documents and explain the reason.
- The 2–3 day target is a working schedule, not a guarantee. Do not reduce mandatory coverage or invent evidence to meet it.

Suggested pacing: Day 1 prompts 00–05; Day 2 prompts 06–10; Day 3 prompts 11–15 plus remaining checks.

## Prompt index

- [Prompt 00: Requirements, provenance, and environment](#prompt-00)
- [Prompt 01: Prove xv6 can run early](#prompt-01)
- [Prompt 02: Shared experiment infrastructure](#prompt-02)
- [Prompt 03: Track B: exact and heuristic discrete methods](#prompt-03)
- [Prompt 04: Track A: continuous methods and foundations](#prompt-04)
- [Prompt 05: Track E: main case study](#prompt-05)
- [Prompt 06: Track F: compiler](#prompt-06)
- [Prompt 07: Track G: actual xv6 optimization](#prompt-07)
- [Prompt 08: Track H: database](#prompt-08)
- [Prompt 09: Track J: realistic software-engineering task](#prompt-09)
- [Prompt 10: Track D: submodular and parallel experiments](#prompt-10)
- [Prompt 11: Track D: distributed and multi-task optimization](#prompt-11)
- [Prompt 12: Track C: quantum optimization](#prompt-12)
- [Prompt 13: Track I: quantum algorithms and QML](#prompt-13)
- [Prompt 14: Audit, reproduce and write the report](#prompt-14)
- [Prompt 15: Demonstration and repository handoff](#prompt-15)

<a id="prompt-00"></a>

## Prompt 00 — Requirements, provenance, and environment

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 00: Requirements, provenance, and environment.

Actions: Inspect available Python, Git, compilers, QEMU, RISC-V toolchain, storage and CPU resources. Select one supported Python environment and a small dependency set: NumPy/SciPy, plotting, pytest, an ILP interface, PyTorch, and one quantum framework. Verify current official installation instructions when installing. Prefer CPU-only experiments. Record versions and actual commands; do not assume this machine has working dependencies.

Create the status record, environment notes, AI register, prompt-log format and a provisional Chapter 6 coverage matrix. Preserve this planning interaction as AI assistance. Initialize Git if needed, with AI-assisted commit labels such as `[AI-assisted]`; distinguish subsequent manual work. Never guess the model version if unavailable—record it as unavailable.

Acceptance checks: Environment smoke tests run, installation blockers are recorded, every assignment requirement has a planned experiment, and AI provenance is being captured. Request class notes without blocking independent implementation.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-01"></a>

## Prompt 01 — Prove xv6 can run early

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 01: Prove xv6 can run early.

Actions: Obtain xv6-riscv from its official source, record upstream commit and license, build and boot it under QEMU. Record the toolchain, QEMU version, CPU count and invocation. Run baseline `usertests` and save output. Select a scheduling modification compatible with that source version; avoid carrying a patch from an unrelated xv6 version.

Acceptance checks: xv6 boots and baseline tests pass, or a concrete environment blocker is recorded. Do not defer a broken toolchain until the report stage. If blocked, continue steps 02–06 while fixing access/toolchain separately; Track G remains incomplete.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-02"></a>

## Prompt 02 — Shared experiment infrastructure

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 02: Shared experiment infrastructure.

Actions: Implement configuration, seed handling, metadata capture, raw output storage and plot generation. Record hardware/OS, dependency versions, Git revision and working-tree changes, input hashes, solver settings and measurement units. Separate setup/compilation from execution timing, and report both where relevant.

Use small debug runs first, then an explicitly configured final run. Target at least five seeds for stochastic comparisons where feasible; report individual runs and mean/std or median/IQR. Use repeated timings for short deterministic tasks. State justified exceptions and timeouts. Never compare a tiny baseline instance with a different optimized instance.

Acceptance checks: A smoke experiment saves and reloads valid raw results; plots are generated from these results; invalid input and failed runs are represented honestly.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-03"></a>

## Prompt 03 — Track B: exact and heuristic discrete methods

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 03: Track B: exact and heuristic discrete methods.

Actions: Generate seeded knapsack cases. Compare value/weight greedy, DP, ILP and an explicit Branch-and-Bound implementation. Use exhaustive enumeration on tiny instances to validate feasibility and optimal values.

Create a small constrained binary selection problem with a justified penalty, for example a fixed-cardinality selection objective. Derive its QUBO, spin conversion and constant offset. Verify every bitstring on tiny instances, show feasible optimal solutions at a sufficient penalty, and demonstrate how an insufficient penalty can fail. Export the identical coefficients and instances for Track C. Include feature selection if class notes require it separately.

Acceptance checks: Independent checks pass; exact/heuristic gaps and runtimes are saved; QUBO and Ising energies agree including offsets; the shared quantum input is frozen.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-04"></a>

## Prompt 04 — Track A: continuous methods and foundations

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 04: Track A: continuous methods and foundations.

Actions: Implement or transparently use libraries for GD, SGD, mini-batch SGD, Momentum, NAG, AdaGrad, RMSProp and Adam on suitable tasks. Use Newton on a smooth small objective and Gauss–Newton on nonlinear least squares; they need not share an unsuitable benchmark.

Demonstrate parameters/loss/regularization, continuous versus discrete variables, convex/strongly convex/non-convex functions, a saddle and local/global minima. Show backpropagation separately from the optimizer, with finite-difference gradient checks. Include Hessian-vector products, a diagonal curvature approximation and Fisher-based curvature; add low-rank treatment if taught. Examine step sizes, noisy gradients/objectives and empirical convergence. State assumptions before mentioning theoretical rates.

Acceptance checks: Known-optimum and gradient checks pass, all named methods have real runs, and convergence/quality/runtime/sensitivity plots are generated. Use comparable gradient-evaluation or time budgets and explain second-order costs.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-05"></a>

## Prompt 05 — Track E: main case study

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 05: Track E: main case study.

Formulation: For job j, choose start time s_j and worker assignment. Minimize C_max. Enforce s_j >= release_j, predecessor completion before successor start, at most one active job per worker, and C_max >= s_j + duration_j. Explain any discretization, horizon, and big-M bounds in the ILP.

Actions: Measure a fixed-order list scheduler. Compare a critical-path or longest-ready-job heuristic and an ILP on identical DAG instances. Use small instances for optimality verification and larger ones for scaling with solver time limits. A timed-out feasible solution is not a proven optimum: preserve status, bound and gap.

Acceptance checks: An independent validator checks every schedule. Record makespan, runtime, feasibility and lower-bound/optimality gaps. Include a strategy table covering variable types, constraints, size, exact/heuristic choice, hardware and parallelism. Explain why quantum/gradient methods are not automatically appropriate here.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-06"></a>

## Prompt 06 — Track F: compiler

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 06: Track F: compiler.

Actions: Build a small toy language with lexer, parser, IR and code generator targeting executable bytecode or another documented target. Implement constant folding, constant propagation, dead-code elimination and common-subexpression elimination with explicit semantics. Specify integer arithmetic and side effects.

Formulate a genuine compiler decision, such as choosing a pass sequence under a compile-time budget; solve it with a named greedy or DP method and compare with a fixed order. Keep tuning programs separate from evaluation programs. Do not label the compiler's ordinary implementation as the required optimization formulation.

Acceptance checks: Source execution/reference interpretation, unoptimized generated code and optimized code agree on known and generated programs. Include reassignment/side-effect edge cases. Save static/dynamic instruction counts, executable size, execution time and compile time with units.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-07"></a>

## Prompt 07 — Track G: actual xv6 optimization

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 07: Track G: actual xv6 optimization.

Actions: Implement one bounded kernel feature, preferably a scheduling policy with well-defined accounting. Add only the user-visible controls/statistics necessary for its experiment. Document locking, process-state transitions and fairness implications. Run tests after every kernel change.

Create at least three user workloads: CPU-bound, mixed CPU/sleep, and latency-sensitive/background competition. Compare baseline and modified kernels with the same QEMU configuration and workload parameters; use one emulated CPU for the first scheduling comparison. Record scheduling metrics with their clock resolution. Add a second independent feature only after the first passes; document whether it meets the full-credit target.

Acceptance checks: Modified xv6 boots, passes `usertests`, preserves workload outputs and produces before/after measurements for all three workloads. Submit a reproducible diff against the pinned upstream revision, benchmark sources and full logs. Explain limitations of QEMU timings.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-08"></a>

## Prompt 08 — Track H: database

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 08: Track H: database.

Formulation: Choose binary index variables x_i, maximize measured workload benefit or minimize latency, subject to measured index storage <= budget. Explain that additive benefit estimates can miss interactions and validate the selected set by executing the complete workload.

Actions: Generate a documented relational SQLite dataset, starting small and scaling to a useful size (for example 100,000 rows in the main table). Create a multi-query workload and several candidate indexes. Measure the default/no-added-index baseline, greedy selection and ILP or DP selection. Include an all-index reference if useful.

Acceptance checks: Preserve `EXPLAIN QUERY PLAN` output, repeated query latencies, workload throughput and storage for each selected set. Compare results with an ordering-aware or multiset-aware checker. Document warm-cache conditions and index-build costs; do not call SQLite plans `EXPLAIN ANALYZE` results.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-09"></a>

## Prompt 09 — Track J: realistic software-engineering task

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 09: Track J: realistic software-engineering task.

Actions: Select a public, licensed, pinned codebase with at least 1,000 meaningful source lines, a runnable test suite and usable coverage. Record actual source size, license and provenance; do not inflate line count. Profile the full suite and gather per-test coverage and durations.

Formulation: Choose tests to minimize total measured runtime while preserving the target coverage set. Compare full-suite, deterministic naive coverage-preserving selection, greedy set cover and ILP. All reduced-suite comparisons must use the same coverage requirement. Separate selection/profiling overhead from execution savings.

Acceptance checks: Run selected suites, recompute retained coverage, and report runtime/selection overhead, test counts and maintainability implications. Coverage preservation does not establish equal fault detection: use a small seeded-mutation check if feasible, otherwise state that limitation. Document why repeated CI runs might amortize profiling cost.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-10"></a>

## Prompt 10 — Track D: submodular and parallel experiments

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 10: Track D: submodular and parallel experiments.

Actions: Use maximum coverage under a cardinality budget. Check normalization, non-negativity, monotonicity and diminishing returns by exhaustive small-set tests. Compare greedy with exhaustive optimum on small cases. State the conditions behind the 1-1/e guarantee and do not transfer it to unrelated objectives.

For actual parallel execution, compare serial, static balanced assignment and dynamic task distribution/work stealing. A centralized shared queue is dynamic scheduling, not work stealing; implement per-worker deques and actual stealing if claiming the latter. Include uneven task lengths, locality and synchronization/communication costs. Demonstrate data, model and pipeline parallelism with small real process-based computations; measure overhead even if they are slower.

Acceptance checks: Outputs agree, multiple worker counts are tested within available cores, and speedup S_p=T_1/T_p and efficiency S_p/p are reported. Use fixed-size and scaled-size experiments to discuss Amdahl and Gustafson respectively. Label any assumptions or fitted serial fractions.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-11"></a>

## Prompt 11 — Track D: distributed and multi-task optimization

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 11: Track D: distributed and multi-task optimization.

Actions: Compare centralized learning and FedAvg on fixed synthetic or documented public data. Vary local updates and communication rounds, include an IID/non-IID split, weight client updates by sample count, and report both local compute and transmitted parameter bytes. Distinguish same-total-update comparisons from communication-budget comparisons.

Train a small shared model with two task heads. Compare equal weighting, MGDA and GradNorm with the same split, initialization seeds and training budget. Validate MGDA's constrained weighting on a known two-gradient case. Ensure GradNorm's learned weights receive the intended gradient and normalization; do not substitute a heuristic while retaining the algorithm's name.

Acceptance checks: Save per-task metrics, communication rounds/bytes, losses and runtime; include implementation checks and explain task trade-offs. Explicitly label local simulation rather than geographically distributed hardware.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-12"></a>

## Prompt 12 — Track C: quantum optimization

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 12: Track C: quantum optimization.

Actions: Load Track B's exact QUBO and construct its cost Hamiltonian with tested basis/sign conventions. Run QAOA and compare expected objective, feasible-sample rate and best sampled feasible solution with the classical optimum. Run small VQE against exact diagonalization and a small trainable PQC/QNN. Demonstrate parameter-shift, SPSA and metric-based quantum natural gradient.

Use analytic versus finite-shot measurements, several shot budgets, and a documented hardware noise model where supported. Measure gradient variance over seeds as qubits/depth increase. A small decreasing-variance experiment is evidence of a trend, not proof of a general barren-plateau law. Account for the shot/circuit cost of gradients and metric estimation.

Acceptance checks: Circuit and derivative checks pass; raw seeds, shots, circuit calls and optimizer budgets are saved; noisy results are labelled simulated. Include basic quantum concepts and the role of optimization in QML.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-13"></a>

## Prompt 13 — Track I: quantum algorithms and QML

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 13: Track I: quantum algorithms and QML.

Actions: Provisionally cover Deutsch–Jozsa, Bernstein–Vazirani, Grover, QFT, QPE, a small Shor demonstration and quantum annealing; reconcile this list with actual class material. Write circuits for at least two algorithms directly and test known outputs/distributions. If Shor is compiled for a fixed instance, disclose the simplification and do not present it as general factoring. Classical simulated annealing is a classical baseline, not a quantum-annealing implementation; use a valid small quantum annealing evolution/model and discuss embedding limits.

Compare Grover and linear search using matched oracle assumptions, query counts, gate counts and simulator wall time. Train a variational classifier/QNN and a classical model on the same train/validation/test data; also demonstrate a quantum kernel if taught. Match tuning effort and prevent data leakage. Include a noisy execution and reuse the documented noise model from C if appropriate, linking distinct outputs.

Acceptance checks: Save accuracy, training runtime, parameter count and circuit evaluations, algorithm verification evidence and all taught-method coverage. Discuss state preparation/data loading, simulation overhead, noise, embedding connectivity and chain overhead for annealers, and compare these limits with actual parallel results from D. Real hardware is optional when access is unavailable; simulated noise must be explicit.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-14"></a>

## Prompt 14 — Audit, reproduce and write the report

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 14: Audit, reproduce and write the report.

Actions: Reconcile COVERAGE.md with the exact syllabus if obtained. Mark each item demonstrated, missing or unconfirmed; a description alone is not coverage. Run a clean-install smoke workflow and the documented final commands. Check that figures and claims trace to raw results and that each track has a correctness check.

Write `docs/REPORT.md` with an A–J section. Each section must include process, mathematical formulation, baseline, method rationale, setup, correctness evidence, results with units/variability, limitations and code/result links. Answer all ten final assignment questions. Include honest unsuccessful optimizations and why they failed. Export to PDF only after content is verified if a PDF is desired/required.

Acceptance checks: All mandatory items have evidence or are explicitly disclosed as incomplete; report tables match raw files; setup commands work; licenses and AI disclosures are complete. No unchecked step may silently become “done.”

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

<a id="prompt-15"></a>

## Prompt 15 — Demonstration and repository handoff

```text
Read README.md, Prompt.md, docs/STATUS.md if present, and applicable AGENTS.md instructions. Inspect existing files and results before making changes. Execute the specified step and necessary prerequisites only; do not start the next numbered step automatically. Implement, run and independently verify the work rather than only proposing code.

Use the step specification below together with README.md. Keep the experiment small but satisfy every listed requirement. Use actual measurements, deterministic configurations and repeated runs where relevant. Preserve raw outputs, seeds, environment details, solver status and meaningful correctness checks. Do not invent results, AI mistakes, screenshots, model identifiers or completed checks. Record missing information and blockers explicitly. Ask only for information or authorization that is genuinely needed; continue independent authorized work when possible.

Save this actual prompt and relevant response excerpts in prompts/<track>/ or prompts/setup/, with date, tool/model if known, changed files, verification notes and commit references when available. Record genuine AI failures as they occur. Update README.md progress and docs/STATUS.md, including commands run, checks passed, unresolved issues and the exact next action. Mark a step complete only when its acceptance checks pass; otherwise use IN PROGRESS or BLOCKED with a reason. Do not publish externally without authorization.

Execute step 15: Demonstration and repository handoff.

Actions: Prepare a short demonstration script showing formulation, one before/after comparison, independent correctness checks, xv6 boot/benchmark evidence, quantum limitations and AI corrections. Include viva questions and small code modifications the student can practice without AI.

Review files for secrets, private data, unnecessary binaries and license compliance. Ensure the public README includes setup, execution, results navigation, AI tools table and external-source acknowledgments. Verify the remote and obtain explicit publication authorization before creating/pushing a public repository if not already authorized.

Acceptance checks: Public repository URL is verified, clean-clone instructions are tested, report is accessible, and the student can explain the code. Never fabricate a publication URL or mark publication complete after a local commit alone.

Finish with: completion status, key files, measured findings (if any), verification commands/results, blockers, and the next prompt number.
```

## Additional required task — actual second AI tool

After setup, use the following identical task in Codex and one other AI tool you can actually access. Keep original outputs separate. This is a comparison task, not an extra replacement for Track B. Save genuine exports or screenshots where available. Do not send project secrets to either tool.

```text
Implement a small Python 0/1 knapsack solver using dynamic programming, plus a value-to-weight greedy baseline. Inputs are non-negative integer values, strictly positive integer weights and a non-negative integer capacity. Each item may be selected at most once. Return selected original indices and total value/weight. Define behavior for invalid inputs and ties. Include a separate brute-force verifier for small inputs and tests for empty inputs, zero capacity, duplicate items, zero-value items, overweight items and a case where greedy is suboptimal. Explain complexity and independently verify returned selections and objective values. Use only the standard library. Do not claim tests were run unless you actually executed them.
```

After genuine outputs from both tools are available, run this comparison prompt:

```text
Read README.md, Prompt.md and docs/STATUS.md. Locate the genuine saved outputs for the identical second-AI-tool knapsack task. If either is absent, record the missing evidence and do not fabricate it. Preserve each original implementation and test both with the same independently written verifier and inputs. Compare correctness, unsupported claims, hallucinations, prompt iteration counts and response timing only where actually recorded. Document revisions separately from original outputs. Write docs/AI_TOOL_COMPARISON.md and update the README AI tools register and real prompt logs. Record actual failures and fixes without treating deliberately seeded errors as AI mistakes. Do not start the next experiment automatically.
```

## Resume an interrupted step

```text
Read README.md, Prompt.md, docs/STATUS.md and applicable AGENTS.md instructions. Find the earliest IN PROGRESS step, or the earliest TODO step if none is in progress. Inspect existing changes, results and logs. State the step number, then continue from the recorded next action without recreating completed work or overwriting evidence. If it is blocked, explain the exact dependency and continue only independent authorized tasks. Finish the step's checks and update status and authentic prompt logs. Stop after this step and report the next prompt.
```

## Final check before submission

```text
Audit the repository against README.md and the original assignment requirements. Verify all ten tracks, exact-syllabus coverage status, genuine raw measurements, reproducibility, report links, xv6 patch and usertests logs, compiler equivalence, SQL result equivalence, quantum/classical distinctions, parallel/distributed measurements, real AI failure logs and the actual two-tool comparison. Check every completion claim against evidence. List gaps explicitly and correct unsupported claims. Do not silently mark missing work as done. Confirm publication only from a verified public repository URL; prepare local artifacts without publishing unless already authorized.
```

**Current next action: obtain genuine second-AI-tool evidence and complete user-managed Git/publication. Prompt 15 was explicitly declined. Prompts remain specifications; only verified progress records establish completion.**
