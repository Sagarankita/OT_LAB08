# Track I — Quantum algorithms and QML comparison

## Scope and execution model

This track runs entirely on local PennyLane simulators. It contains no result from physical quantum hardware. Ideal statevector results, finite-shot sampling, and simulated depolarizing noise are labelled separately. The selected run is `results/track_i_quantum/20261029T180846.017650Z/`.

The experiment covers Deutsch–Jozsa (DJ), Bernstein–Vazirani (BV), Grover search, the quantum Fourier transform (QFT), quantum phase estimation (QPE), a compiled Shor order-finding example, a unitary quantum-annealing model, a variational quantum classifier (VQC), and a quantum kernel. DJ, BV, and Grover use directly written circuits and oracles rather than algorithm wrappers.

## Algorithm experiments

| Experiment | Formulation / baseline | Measured result | Correctness evidence |
|---|---|---|---|
| DJ | Decide constant versus balanced parity oracle; deterministic classical worst case needs more than one query | Both known cases classified with one oracle query | Shot distributions exclude/include `000` as expected |
| BV | Recover four-bit secret `1011`; deterministic classical baseline uses four oracle queries | Secret recovered with one oracle query | Exact known-answer bitstring |
| Grover | Find marked value `101` among eight candidates; linear scan is the classical baseline | 2 oracle calls, success probability 0.9453125; fixed scan used 6 calls, uniform expected 4.5, worst case 8 | Direct oracle/diffusion test and 1,000-shot sample |
| QFT | Transform basis state `101` and invert it | Analytic-state fidelity 1.0; round-trip probability 1.0 | Compared with the normalized discrete Fourier state |
| QPE | Estimate the eigenphase 3/8 with three estimation qubits | Estimated 3/8 with probability approximately 1.0 | Exact representable phase known answer |
| Compiled Shor | Infer the order of 2 modulo 15 from QPE, then use gcd post-processing | Order 4 and factors 3, 5 | Non-zero useful phase probability 0.75; explicit permutation-unitary check |
| Quantum annealing | Minimize a three-spin antiferromagnetic Ising chain using time-dependent closed-system evolution | Ground-state success rose from 0.2890 at T=0.5 to 0.9946 at T=8 | Every output distribution sums to one and energies are independently enumerated |

The Shor demonstration compiles the fixed modular-multiplication unitary for N=15 and a=2. It is an order-finding demonstration, not a scalable or general factoring implementation. The annealing experiment uses first-order Trotterized gate-model evolution from a transverse-field driver to an Ising problem Hamiltonian. It is not classical simulated annealing and it is not execution on a commercial annealer.

Grover demonstrates query complexity, not a wall-clock speedup. The local simulator averaged 0.001019 s, while the tiny classical scan averaged 2.23e-7 s. Simulation explicitly tracks amplitudes and incurs Python/framework overhead, so it was about 4,563 times slower in this measured three-qubit case despite using fewer oracle calls.

## Matched QML experiment

The shared dataset has 40 deterministic two-dimensional moon-like examples: 24 train, 8 validation, and 8 test. The split is identical for every model. Mean and standard deviation are fitted on training data only, then features are clipped and angle-scaled. Hyperparameters are selected with validation data; the test split is evaluated only after selection.

The VQC has two qubits, three trainable layers, six parameters, angle encoding, entangling CNOTs, and a Pauli-Z output. Parameter-shift gradients and Adam optimize mean-squared error. The matched classical baseline is a 2–3–1 MLP with 13 parameters and the same three seeds and 40-epoch budget. Kernel ridge classification compares the quantum fidelity kernel with a classical RBF kernel on the same split.

| Model | Test accuracy | Training/runtime | Parameters | Circuit evaluations |
|---|---:|---:|---:|---:|
| VQC, ideal simulator, mean of 3 seeds | 1.000 | 6.658 s mean | 6 | 12,520 mean |
| Classical MLP, mean of 3 seeds | 1.000 | 0.00859 s mean | 13 | 0 |
| Quantum kernel | 1.000 | 0.499 s | no trained circuit weights | 684 |
| Classical RBF kernel | 1.000 | 0.000701 s | n/a | 0 |
| VQC, p=0.02 simulated depolarizing noise, 1,000 shots | 1.000 mean | test inference only | 6 | 8 per seed |

The equal accuracies are a negative advantage result: this small separable dataset does not demonstrate a QML benefit. The ideal VQC is roughly 775 times slower to train than the MLP and uses thousands of circuit evaluations. The noisy result also should not be generalized: eight test examples, a simple noise model, and 1,000 shots have low statistical power.

## Costs and limitations

- State preparation is treated as angle encoding. Loading a general classical vector can dominate an end-to-end quantum workflow and is not included in the reported model runtimes.
- Finite shots add sampling error; simulated depolarizing channels approximate only one class of hardware noise. There are no calibration, queue, readout, or connectivity costs from real hardware.
- The quantum-kernel Gram matrix required 684 circuits even for 40 samples; kernel evaluation scales quadratically with sample count.
- Real annealers have sparse hardware graphs. Embedding logical edges can require chains of physical qubits, chain-strength tuning, and broken-chain decoding; the three-spin simulation pays none of those costs.
- Track D measured a real 1.565× four-process speedup with 0.391 efficiency for suitable CPU work. That is an observed wall-clock gain on this machine. Grover's advantage here is only an oracle-query comparison, and simulator timing was worse than classical timing.
- The exact Chapter 6 class notes remain unavailable, so this is coverage of the algorithms named in the assignment; Prompt 14 must reconcile any additional taught variants.

## Reproduce and verify

```sh
.venv/bin/python -m pytest -q tests/test_quantum_algorithms.py
.venv/bin/python scripts/run_track_i.py --config configs/track_i_final.json
.venv/bin/python scripts/verify_track_i.py
```

The independent verifier passed 41/41 checks and re-executed DJ, BV, Grover, and annealing circuits rather than trusting saved labels. The complete project suite passed 68/68 tests.

Key files: `src/optimization_lab/quantum_algorithms.py`, `scripts/run_track_i.py`, `scripts/verify_track_i.py`, `tests/test_quantum_algorithms.py`, and the selected result directory above.
