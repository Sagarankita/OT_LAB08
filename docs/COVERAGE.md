# Requirement coverage audit

Audited 6 November 2026. Every method explicitly named in the assignment has a demonstrated experiment, but the exact Chapter 6 class notes remain unavailable. Therefore course-specific methods not named in the assignment cannot be certified. The detailed evidence map is in `docs/REPORT.md`; reconcile any additional taught method before submission.

| Requirement | Track | Step | Planned evidence | State |
|---|---|---|---|---|
| ML parameters, loss, regularization; continuous/discrete | A/B | 04/03 | Ridge formulation plus continuous/integer example and knapsack | DEMONSTRATED |
| Convex, strongly convex, non-convex; local/global minima and saddles | A | 04 | Analytic landscapes, multiple starts and derivative/curvature checks | DEMONSTRATED |
| GD; SGD; mini-batch SGD; Momentum; NAG; AdaGrad; RMSProp; Adam | A | 04 | Five seeds, equal data exposure, reproduced tuning and 40 measured records | DEMONSTRATED |
| Newton; Gauss–Newton | A | 04 | Exact quadratic and nonlinear least-squares/SciPy verification | DEMONSTRATED |
| Backpropagation distinct from optimizer | A | 04 | Manual XOR gradient vs finite differences/PyTorch; Adam training | DEMONSTRATED |
| Fisher; diagonal/low-rank curvature; Hessian-vector products | A | 04 | Expected/empirical Fisher, approximation errors and finite-difference HVP | DEMONSTRATED (exact syllabus pending) |
| Convergence; step sizes; noisy objectives | A | 04 | Convexity classes, stability boundary, optimizer tuning and five noisy seeds | DEMONSTRATED |
| Greedy; ILP; explicit Branch-and-Bound; DP | B | 03 | Five seeded knapsacks, 100 records, exhaustive oracle and independent verifier | DEMONSTRATED |
| Feature selection/combinatorial application | B | 03 | Knapsack plus fixed-cardinality project selection; exact syllabus pending | DEMONSTRATED (provisional syllabus) |
| QUBO penalty terms; binary-to-spin conversion | B/C | 03/12 | All 32 classical and Hamiltonian energies checked; exact handoff used by QAOA | DEMONSTRATED |
| Basic quantum concepts; optimization in QML | C/I | 12/13 | Variational circuits, measurement/shot/noise experiments and same-data algorithm/QML comparisons | DEMONSTRATED |
| QAOA; VQE; PQC/QNN; hybrid methods | C | 12 | Exact Track B Hamiltonian, QAOA sampling, VQE eigensolution and trained QNN | DEMONSTRATED |
| Parameter-shift; SPSA; quantum natural gradient | C | 12 | Finite-difference derivative, call counts and Fubini–Study metric | DEMONSTRATED |
| Barren plateaus; finite shots; noise | C | 12 | 2/4/6-qubit variance grid, 100/1k/5k shots and depolarizing simulation | DEMONSTRATED |
| Deutsch–Jozsa; Bernstein–Vazirani; Grover | I | 13 | Direct circuits, known-answer tests, query/gate counts and matched linear-search comparison | DEMONSTRATED |
| QFT; QPE; Shor | I | 13 | Analytic QFT/QPE checks and explicitly compiled N=15 Shor order finding | DEMONSTRATED |
| Quantum annealing; QUBO hardware embedding limits | I | 13 | Unitary time-dependent Ising evolution, normalization checks and connectivity/chain limits | DEMONSTRATED |
| Variational classifier; quantum kernels | I | 13 | Leakage-free same-data VQC/MLP and quantum/RBF-kernel comparisons | DEMONSTRATED |
| Diminishing returns; greedy assumptions/guarantees | D | 10 | 128 subsets, 5,103 diminishing-return comparisons, ten greedy/exact instances | DEMONSTRATED |
| Load balance; makespan; work stealing; locality; synchronization/communication | D | 10 | Real processes, centralized queue vs shared-memory deques, 1/2/4 workers and output hashes | DEMONSTRATED |
| Data/model/pipeline parallelism; speedup/efficiency; Amdahl/Gustafson | D | 10 | Fixed/scaled runs, process-based paradigms, overhead, formulas and 31-check audit | DEMONSTRATED |
| FedAvg; local steps/communication rounds | D | 11 | Unequal IID/non-IID clients, same-compute/communication views, bytes and 30-check audit | DEMONSTRATED |
| MGDA; GradNorm vs equal task weighting | D | 11 | Same-seed two-task model, known MGDA cases and differentiated GradNorm weights | DEMONSTRATED |
| Strategy selection and main case study formulation | E | 05 | 15 DAGs, fixed/critical-path/MILP comparison, exhaustive small oracle and 17-check validator | DEMONSTRATED |
| Compiler pipeline; four passes; formulated decision | F | 06 | Lexer/parser/IR/bytecode, four passes, DP pass order, 270 records and independent equivalence | DEMONSTRATED |
| Actual xv6 kernel feature; three workloads; boot/usertests; patch | G | 01/07 | Weighted stride, 48 paired records, three workloads, 74 usertests and dry-run patch audit | DEMONSTRATED (minimum one-feature scope) |
| Second xv6 feature (full-credit target) | G | 07 | Not implemented; explicitly disclosed | NOT IMPLEMENTED |
| DBMS budgeted selection; two methods; plans/latency/throughput/storage | H | 08 | 100k-order SQLite, greedy/DP, 168 timings, plans, ordered hashes and storage audit | DEMONSTRATED |
| Non-trivial codebase; profiling; named optimization; correctness/cost | J | 09 | Pinned 4,728-line more-itertools, 671 tests, naive/greedy/optimal ILP, coverage replay, mutations and 22-check audit | DEMONSTRATED |
| Repeated measurements; raw data/plots; reproducibility | All | 02/14 | Shared schema, per-track raw artifacts, 68 tests, dependency audit and all selected verifiers rerun | DEMONSTRATED |
| Every taught method; exact Chapter 6 reconciliation | All | 00/14 | Every assignment-listed method is demonstrated; additional class-note-only methods cannot be checked because notes are unavailable | UNCONFIRMED |
| AI register; prompts/exports; traceability; 3 genuine failures; 2 tools | All | 00–15 | Codex register, prompt excerpts, 11 failures and final snapshot commit exist; second tool, full exports and per-step commit links remain unavailable | PARTIAL |
| Public repository; report A–J; demo; ten final questions | All | 14/15 | Public repository, detailed A–J report and ten answers complete; demo explicitly declined | PARTIAL |
