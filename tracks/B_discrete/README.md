# Track B — discrete optimization

## Problem 1: 0/1 knapsack

For item `i`, let binary decision variable `x_i` indicate selection, value `v_i` measure benefit and weight `w_i` consume capacity. The formulation is:

```text
maximize    sum_i v_i x_i
subject to  sum_i w_i x_i <= C
            x_i in {0,1}
```

Five instances were generated from fixed seeds. Each has 18 items with integer value 1–60 and weight 1–30. Capacity is `floor(0.35 * total item weight)`. The same instance and capacity were supplied to every method. Input generation, exhaustive enumeration and one warm-up per method are counted as setup; five timed repetitions per instance and method are stored individually.

The baseline is descending value-to-weight ratio greedy selection. It is inexpensive but has no general optimality guarantee for indivisible 0/1 items. Dynamic programming is exact and pseudo-polynomial in `O(nC)` time. The explicit Branch-and-Bound solver uses a fractional-knapsack upper bound and a greedy incumbent; it is exact but exponential in the worst case. SciPy's MILP interface solves the binary linear model with HiGHS; its status and node count are retained. Exhaustive `2^n` enumeration is an independent tiny-instance oracle, not a scalable proposal.

## Measured result

Selected evidence: `results/track_b_discrete/20261027T133954.808190Z`.

| Method | Mean time | Sample SD | Mean gap | Maximum gap | Correctness |
|---|---:|---:|---:|---:|---|
| Greedy | 0.00644 ms | 0.00031 ms | 0.5260% | 2.0619% | Feasible on every run |
| Dynamic programming | 0.12360 ms | 0.00803 ms | 0% | 0% | Matched exhaustive optimum |
| ILP / HiGHS | 2.52765 ms | 0.08592 ms | 0% | 0% | Matched exhaustive optimum |
| Branch-and-Bound | 0.03367 ms | 0.00270 ms | 0% | 0% | Matched exhaustive optimum |

These timings apply only to the documented 18-item instances on this machine. They do not establish general algorithm rankings: DP depends on numeric capacity, Branch-and-Bound depends on pruning and item order, and ILP carries solver overhead that dominates tiny cases.

Greedy matched the optimum for seeds 101, 404 and 505. It returned 380 versus 388 for seed 202 and 350 versus 352 for seed 303. The known unit-test counterexample with `(value,weight)` items `(60,10), (100,20), (120,30)` and capacity 50 also confirms greedy value 160 versus exact value 220.

## Problem 2: cardinality-constrained project selection

The QUBO experiment chooses exactly two of five projects. Let benefits be `b_i`, pairwise complementarities be `s_ij`, and `S(x)` be:

```text
S(x) = sum_i b_i x_i + sum_{i<j} s_ij x_i x_j
sum_i x_i = k, where k = 2
```

The minimization energy is:

```text
E(x) = -S(x) + P (sum_i x_i - k)^2
```

With binary `x_i`, the expanded upper-triangular QUBO is:

```text
offset = P k^2
linear_i = -b_i + P(1 - 2k)
quadratic_ij = -s_ij + 2P
```

All benefits and complementarities are non-negative. Their total, 39, upper-bounds every reward. Choosing `P = 40` proves feasibility of every global minimizer: an infeasible state incurs at least 40 penalty and therefore has energy greater than zero, while every feasible state has energy `-S(x) <= 0`.

The sufficient-penalty optimum is bits `[1,1,0,0,0]`, score 18 and energy −18. With the deliberately insufficient `P = 1`, global minima include four- and five-project selections with energy −30, demonstrating that an arbitrary small penalty does not enforce the constraint.

For Ising spins `z_i in {-1,+1}`, use `x_i=(1-z_i)/2`. The exported energy is:

```text
E(z) = constant + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j
```

The exact coefficients, offset, convention, classical optimum and penalty proof are frozen in `data/track_b_qubo.json` with SHA-256 `69b36ca819b8dd40718db89d63c01391333716d144b62130dc3bd372cf276dad`. Track C must load this file directly. Enumeration verified original penalized energy, QUBO energy and Ising energy for all 32 states under both penalties.

## Correctness and reproduction

The independent verifier reconstructs all five exhaustive knapsack oracles, checks 100 raw method records, recomputes every reported gap, verifies feasibility and exact-method equality, enumerates both QUBO penalties, compares the project handoff byte-for-byte with the run artifact, and verifies every declared artifact hash. All 19 checks passed.

```sh
.venv/bin/python -m pytest tests/test_discrete.py
.venv/bin/python scripts/run_track_b.py --config configs/track_b_debug.json
.venv/bin/python scripts/run_track_b.py --config configs/track_b_final.json
.venv/bin/python scripts/verify_track_b.py results/track_b_discrete/20261027T133954.808190Z
```

The exact Chapter 6 notes are still unavailable. This experiment covers the assignment-listed greedy, DP, ILP, explicit Branch-and-Bound, combinatorial selection, QUBO penalties and binary-to-spin conversion. Any additional taught discrete method must be added during the syllabus audit.

