# Track D — distributed and multi-task optimization

This completes Prompt 11's FedAvg, MGDA and GradNorm requirements. All distributed-learning results are a deterministic **single-machine local simulation**, not geographically distributed hardware.

## 1. FedAvg formulation and accounting

A synthetic binary logistic-regression problem uses 600 training and 300 test examples with six features. Four clients have unequal sizes `[90, 120, 150, 240]`, making sample-count weighting observable. For client parameters `theta_k` and sample counts `n_k`, each round uses

```text
theta_next = sum_k (n_k / sum_j n_j) theta_k
```

The independent check confirms that one full-batch local step followed by weighted aggregation is numerically equal to one centralized full-data gradient step. A separate unequal-client oracle produces weighted `[1.9,1.9]` versus unweighted `[1.5,1.5]`.

The IID partition randomly mixes examples. The non-IID partition is label-major: in every seed, early clients are 0% positive and the largest client is 100% positive. This is deliberately strong label skew.

Communication counts a 64-bit seven-parameter model downloaded and uploaded by each of four clients per round:

```text
bytes = rounds * 4 clients * 7 parameters * 8 bytes * 2 directions
```

Local compute counts full-batch example evaluations:

```text
examples processed = rounds * local steps * 600 examples
```

### Same-total-compute view

Every configuration performs 30 local full-data-equivalent steps, or 18,000 example evaluations. The centralized comparison also performs 30 full-data steps.

| Split | Local steps | Rounds | Bytes | Accuracy mean ± SD | Gap vs centralized |
|---|---:|---:|---:|---:|---:|
| IID | 1 | 30 | 13,440 | 0.9022 ± 0.0948 | 0.0000 |
| IID | 5 | 6 | 2,688 | 0.9022 ± 0.0948 | 0.0000 |
| IID | 10 | 3 | 1,344 | 0.9033 ± 0.0960 | +0.0011 |
| Non-IID | 1 | 30 | 13,440 | 0.9022 ± 0.0948 | 0.0000 |
| Non-IID | 5 | 6 | 2,688 | 0.9067 ± 0.0991 | +0.0044 |
| Non-IID | 10 | 3 | 1,344 | 0.9056 ± 0.0975 | +0.0033 |

Increasing local work from one to ten steps reduced communication tenfold without reducing mean accuracy in this simple convex experiment. This is not evidence that non-IID data is generally harmless: the model is tiny, all clients participate, aggregation is exact, and the reported standard deviation also reflects three different seeded synthetic datasets.

### Same-communication view

All configurations use ten rounds and therefore 4,480 transmitted bytes. Local steps 1, 5 and 10 process 6,000, 30,000 and 60,000 examples respectively. IID accuracies were 0.9089, 0.9022 and 0.9022; non-IID accuracies were 0.9089, 0.9056 and 0.9078. These are not same-compute comparisons: more local steps intentionally spend more computation under the same communication budget.

## 2. Multi-task formulation

A shared `Linear(6,8)+Tanh` representation feeds two binary task heads. Both tasks depend on shared features with a conflicting sign on feature zero. All methods use the same fixed 500/250 train/test split, the same three initialization seeds, Adam settings and 120 epochs. Task weights are normalized to sum two so gradient scale is comparable.

### Equal weighting

The baseline minimizes `L1 + L2`.

### MGDA

For shared gradients `g1` and `g2`, two-task MGDA solves

```text
minimize ||alpha*g1 + (1-alpha)*g2||²
subject to 0 <= alpha <= 1
```

The closed form is clipped to `[0,1]`, then scaled to weights `[2 alpha, 2(1-alpha)]`. Known checks produce `alpha=0.5` for opposing gradients and `alpha=0.8` for `[1,0]` versus `[0,2]`, with the combined norm no larger than either endpoint.

### GradNorm

GradNorm learns positive weights `w_i`. Each epoch computes unweighted shared-gradient norms, constructs `G_i = w_i ||grad L_i||`, and differentiates

```text
sum_i |G_i - mean(G) * r_i^alpha|
```

with `alpha=1.5` and inverse relative training rates `r_i`. Targets and model gradient norms are detached for the weight update; model parameters then receive the weighted loss using detached updated weights. The verifier checks the analytical task-weight gradient, nonzero learned-weight gradients, positivity and sum-to-two normalization at every epoch.

## 3. Multi-task results

| Method | Task 1 accuracy | Task 2 accuracy | Final weights `(w1,w2)` | Runtime (s) |
|---|---:|---:|---:|---:|
| Equal | 0.8587 ± 0.0023 | **0.8600 ± 0.0174** | (1.000, 1.000) | 0.0324 ± 0.0043 |
| MGDA | 0.8427 ± 0.0389 | 0.8280 ± 0.0223 | (0.311, 1.689), high seed variation | 0.0307 ± 0.0002 |
| GradNorm | **0.8627 ± 0.0061** | 0.8560 ± 0.0174 | (0.883, 1.117) | 0.0354 ± 0.0004 |

Equal weighting was the best balanced result. MGDA is correct on its constrained gradient objective but performed worse on test accuracy and varied strongly by initialization. GradNorm shifted weight toward task 2; it slightly improved task 1 accuracy while slightly reducing task 2. Optimizing instantaneous gradient geometry or training rates does not guarantee better generalization.

## Correctness and limitations

- Three seeds are reported with variability.
- IID/non-IID client manifests, per-round history, local updates, example counts and transmitted bytes are saved.
- Sample-weighted FedAvg formulas and a fresh one-step centralized equivalence check passed.
- MGDA known-gradient cases and constrained minimum passed.
- GradNorm's intended weight gradient and normalization passed; it is not a heuristic carrying the GradNorm name.
- All methods share data, initialization per seed and epoch budget.
- The independent verifier passed 30/30 checks; the complete project test suite also passes.

Limitations include full client participation, full-batch deterministic local training, no network latency, no client dropout/privacy/compression, synthetic data, and a small model. Parameter-byte counts do not include protocol headers or optimizer state.

Selected run: `results/track_d_distributed/20261028T092752.487437Z/`.

```sh
.venv/bin/python scripts/run_track_d_distributed.py --config configs/track_d_distributed_debug.json
.venv/bin/python scripts/run_track_d_distributed.py --config configs/track_d_distributed_final.json
.venv/bin/python scripts/verify_track_d_distributed.py results/track_d_distributed/20261028T092752.487437Z
.venv/bin/python -m pytest
```

Git and publication remain user-managed.
