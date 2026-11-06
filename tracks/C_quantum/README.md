# Track C — quantum optimization

All results use PennyLane local simulators. No real quantum hardware was used. Ideal statevector results, finite-shot sampling and simulated noise are labelled separately.

## Quantum concepts and hybrid optimization

A qubit can be in a superposition of computational basis states. Parameterized gates change complex amplitudes; entangling gates create correlations that cannot generally be represented as independent single-qubit states. Measurement produces classical bitstrings, so expectation values estimated with finite shots have sampling variance. Variational algorithms use a hybrid loop: a quantum circuit evaluates an objective or gradient-related quantity, and a classical optimizer updates circuit parameters.

In quantum machine learning, the trainable circuit is a parameterized model and the measured expectation defines its prediction or loss. Optimization is therefore central, but simulation cost, finite shots, noise, data encoding and trainability can outweigh any proposed advantage. This track makes no quantum-speedup claim.

## 1. Track B QUBO and Hamiltonian convention

The experiment loads `data/track_b_qubo.json` directly and preserves its exact coefficients:

```text
E(x) = offset + sum_i linear_i x_i + sum_{i<j} quadratic_ij x_i x_j
x_i = (1-z_i)/2
H_C = constant I + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
```

For a computational basis bit `0`, `Z` has eigenvalue `+1`; for bit `1`, `-1`. This matches the exported binary-to-spin convention. All 32 bitstrings were checked: QUBO energy, Ising energy and Hamiltonian diagonal agree with maximum error zero. The unique classical optimum is `[1,1,0,0,0]`, energy −18 and feasible score 18.

## 2. QAOA

Depth-three QAOA alternates exact cost phases and `X` mixer rotations from the uniform superposition. Three seeded Adam runs use analytic statevector backpropagation for 200 iterations. The best expected energy was −10.3078 (seed 3), above the variational lower bound −18 and therefore not the exact optimum in expectation. Other starts ended at 1.4317 and 2.5355, showing non-convex initialization sensitivity. Each run used 1,005 simulator device executions because the Hamiltonian terms require multiple measurement groups; runtime was about 3.4–3.6 seconds.

The best parameters were sampled under identical circuits:

| Shots | Mean sampled energy | SD across 3 seeds | Mean feasible rate | Optimal feasible bitstring seen |
|---:|---:|---:|---:|---|
| 100 | −11.2900 | 1.6979 | 0.9700 | all seeds |
| 1,000 | −10.5533 | 0.4884 | 0.9710 | all seeds |
| 5,000 | −10.5499 | 0.1331 | 0.9661 | all seeds |

More shots reduced observed estimator variability here, but do not improve the underlying state. Sampling found the optimum even though the expected energy remained above it; “best sampled solution” and “expected objective” are different metrics.

### Simulated noise

`default.mixed` applies a single-qubit depolarizing channel with probability 0.02 to every qubit after each QAOA layer. At 5,000 shots, mean energy degraded to −3.5109 and mean feasible rate to 0.8531. The optimum was still observed in all three noisy samples. This is a documented noise model, not execution on hardware.

## 3. VQE and optimizer costs

The two-qubit benchmark Hamiltonian is

```text
H = -1.05 Z0 + 0.5 Z1 + 0.2 X0 X1
```

Exact diagonalization gives ground energy −1.5628499608. All variational results respect the ground-state lower bound.

| Optimizer | Iterations | Final energy | Error | Device executions |
|---|---:|---:|---:|---:|
| Parameter-shift Adam | 80 | −1.5622475 | 0.0006024 | 1,442 |
| SPSA | 200 | −1.5625275 | 0.0003225 | 802 |
| Quantum natural gradient | 40 | **−1.5627548** | **0.0000951** | 922 |

SPSA uses two objective QNode evaluations per stochastic gradient plus one final evaluation (401 QNode evaluations). Device executions are 802 because non-commuting Hamiltonian terms require two measurement groups. Quantum natural gradient uses a block-diagonal Fubini–Study metric with `1e-3` diagonal regularization; metric condition numbers ranged from 17.4 to 228.2. Its circuit count includes gradient, metric estimation and recorded objective evaluations.

A direct parameter-shift derivative matches central finite differences with absolute error `3.62e-11`. Each scalar check costs two circuit evaluations; parameter shift scales with the number of trainable gate parameters, whereas SPSA remains two objective evaluations per gradient estimate but is noisy.

## 4. Small trainable PQC/QNN

A two-qubit classifier angle-encodes two real features, applies two trainable entangling layers and measures `Z0`. Four parameters are trained by parameter-shift Adam for 30 epochs on 16 examples and evaluated on eight fixed test examples.

Across three initialization seeds, mean train accuracy was 0.8542 and mean test accuracy 0.9583. Each run required 4,344 device executions and roughly 1.8 seconds. The tiny test set makes this only a functional QNN demonstration, not evidence of quantum learning advantage; Track I performs the required matched classical comparison.

## 5. Gradient variance and trainability

For random hardware-efficient circuits, the derivative of a global `Z`-tensor observable was measured by two-circuit parameter shift over 20 seeds for each qubit/depth pair. At depth five, variance decreased from 0.1703 (2 qubits) to 0.02485 (4) and 0.006989 (6). Depth-one and depth-three results also decreased with qubit count.

This is evidence of a decreasing-gradient-variance trend for this specific ansatz, observable, parameter and small size range. It is not proof of a general barren-plateau law. Twenty seeds, at most six qubits, ideal simulation and a global observable are substantial limitations.

## Correctness and reproducibility

- Preserved the exact Track B artifact and checked its file hash.
- Reproduced all 32 basis energies and the classical optimum.
- Recomputed the selected QAOA and every final VQE energy independently.
- Checked shot totals, probability normalization, feasibility and optimal samples.
- Verified parameter shift, SPSA call accounting and QNG metric records.
- Recomputed every saved gradient variance from raw gradients.
- Independent verification passed 34/34 checks; the full Python suite also passes.

Selected run: `results/track_c_quantum/20261028T093727.588182Z/`.

```sh
.venv/bin/python scripts/run_track_c.py --config configs/track_c_debug.json
.venv/bin/python scripts/run_track_c.py --config configs/track_c_final.json
.venv/bin/python scripts/verify_track_c.py results/track_c_quantum/20261028T093727.588182Z
.venv/bin/python -m pytest
```

Git and publication remain user-managed.
