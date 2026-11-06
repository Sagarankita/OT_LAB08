# Track A — continuous optimization

## Main ML formulation

The comparison uses ridge regression with 128 standardized samples and six continuous parameters. For parameter vector `w`, design matrix `X`, target vector `y`, sample count `n` and regularization `lambda=0.05`:

```text
L(w) = (1/(2n)) ||Xw-y||^2 + (lambda/2) ||w||^2
grad L(w) = X^T(Xw-y)/n + lambda*w
H = X^T X/n + lambda*I
```

The Hessian eigenvalues are all positive (`mu=0.7635`, `L=1.3258`, condition number 1.7364), so this experiment is strongly convex and has one analytically computed global optimum. The independent verifier resolves the normal equations and checks the gradient with central finite differences.

Five seeds control initial parameters and stochastic order. Every optimizer sees the same dataset and the same initialization within a seed. Each receives 80 complete data exposures, or 10,240 example-gradient evaluations. This holds data work constant, although update counts differ: GD updates once per epoch, SGD once per example, and the other methods use batches of 16.

## Optimizer tuning

Before measured runs, each method evaluates three learning rates—one quarter, equal to and four times the configuration's center—using fixed seed 999 for 20 epochs. The rate with the lowest final tuning loss is selected. Tuning is recorded as setup and is excluded from method runtime. The verifier reproduces all 24 tuning runs and confirms that measured records use the selected rates.

Selected rates: GD 0.4, SGD 0.0025, mini-batch SGD 0.05, Momentum 0.0075, NAG 0.0075, AdaGrad 0.6, RMSProp 0.04 and Adam 0.12.

## First-order results

Selected evidence: `results/track_a_continuous/20261027T134547.158590Z`.

| Method | Mean runtime | Mean final objective gap | Sample SD of gap |
|---|---:|---:|---:|
| GD | 1.009 ms | 0 (floating-point resolution) | 0 |
| SGD | 29.139 ms | 0.00000648 | 0.00000322 |
| Mini-batch SGD | 2.718 ms | 0.00001260 | 0.00000699 |
| Momentum | 3.013 ms | 0.00002574 | 0.00002790 |
| NAG | 3.276 ms | 0.00002366 | 0.00002521 |
| AdaGrad | 3.309 ms | 0.00017916 | 0.00010418 |
| RMSProp | 3.628 ms | 0.00378071 | 0.00339822 |
| Adam | 4.524 ms | 0.00123614 | 0.00086631 |

All methods reduced loss for every seed. Full-batch GD performed best because this objective is small, deterministic and well-conditioned; this result does not imply that GD generally dominates adaptive methods. SGD processed the same examples but made many more Python-level updates, explaining its larger runtime. RMSProp and Adam retained larger gaps under this narrow three-point tuning budget. Wider tuning, decay schedules or a noisier/non-convex model could change the ranking.

## Foundations and convergence

- **Continuous versus discrete:** `(x-2.3)^2` has continuous optimum `x=2.3`, value 0; restricting `x` to integers changes the optimum to `x=2`, value 0.09.
- **Convex but not strongly convex:** `0.5*x1^2` has Hessian eigenvalues 0 and 1. GD reduces objective, but a flat second coordinate remains unchanged, so distance to one arbitrarily chosen minimizer need not converge.
- **Strongly convex:** `0.5*(x1^2+10*x2^2)` has a unique minimizer and geometric convergence under a stable step.
- **Step-size sensitivity:** for smoothness `L=10`, GD is stable for `0 < alpha < 2/L`. Rates 0.01, 0.10 and 0.19 converge at different speeds; 0.21 diverges in the saved experiment.
- **Non-convex local/global minima:** `x^4-x^2+0.1x` has a global minimum, a higher local minimum and a local maximum. Multiple starts converge to different basins.
- **Saddle point:** `x^2-y^2` has zero gradient at the origin and Hessian eigenvalues −2 and 2. An exact start stays fixed; a small perturbation along negative curvature moves away.

These examples demonstrate behavior rather than prove a universal rate. The usual convex convergence claims require smoothness and step-size assumptions; non-convex optimization generally targets stationary points, which may include saddles or non-global local minima.

## Backpropagation and optimizer distinction

A manually differentiated 2–3–1 XOR network has 13 parameters, binary cross-entropy and L2 regularization. Backpropagation computes its gradient; Adam then uses that gradient to update parameters. The manual gradient matched central finite differences with relative error `5.02e-10` and PyTorch autograd with reported relative error 0 at saved precision. Manual-backprop/Adam training reached 100% accuracy. This separates gradient computation from the optimization rule.

## Second-order and curvature experiments

- **Newton:** on a positive-definite quadratic with constant Hessian, one exact linear solve reached the analytical optimum; final distance was 0 at saved precision.
- **Gauss–Newton:** fitted `a*exp(b*x)` by nonlinear least squares using `J^T J`. Final parameters were within `1.31e-10` of an independent SciPy least-squares solution. This is a local approximation and may fail with poor initialization, rank-deficient Jacobians or large residual effects.
- **Hessian-vector product:** exact `Hv` matched a finite difference of gradients with relative error `3.29e-11`, demonstrating curvature access without forming a new finite-difference Hessian.
- **Fisher:** logistic-regression expected Fisher plus L2 curvature matched the Hessian with relative error `4.64e-17`. The empirical Fisher differed by 0.1076 because it uses observed per-example gradient outer products.
- **Approximations:** the diagonal Hessian relative error was 0.2127; the rank-2 approximation error was 0.6271. The low-rank approximation was worse here, an honest negative result showing that reduced curvature storage can lose important directions.

## Noisy objective

For noisy gradients of `0.5*(x-3)^2`, five seeds were run for 200 iterations. Constant-step final absolute error was `0.0607 ± 0.0295`; the decaying step obtained `0.0181 ± 0.0154`. This small experiment supports variance reduction from decay under its chosen noise and horizon, not a universal dominance claim.

## Verification and reproduction

The independent verifier checks 28 conditions: paired methods, equal data budgets, analytical optimum, saved losses, tuning reproduction, finite-difference gradients, landscape behavior, stable/divergent steps, Newton/Gauss–Newton references, backprop/autograd agreement, XOR accuracy, HVP/Fisher identities, noisy objectives, artifact hashes and honest Git provenance. All passed. The full project suite contains 31 passing tests.

```sh
.venv/bin/python -m pytest
.venv/bin/python scripts/run_track_a.py --config configs/track_a_debug.json
.venv/bin/python scripts/run_track_a.py --config configs/track_a_final.json
.venv/bin/python scripts/verify_track_a.py results/track_a_continuous/20261027T134547.158590Z
```

The exact Chapter 6 notes remain unavailable. This track covers the continuous methods explicitly listed in the assignment; any additional course-specific curvature or convergence method must be reconciled during the final syllabus audit.

