"""Run Track A continuous-optimization experiments and save raw evidence."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from optimization_lab.benchmarking import (
    BenchmarkConfig, RunRecorder, canonical_json_bytes, numeric_summary,
    record_invalid_attempt,
)
from optimization_lab.continuous import (
    FIRST_ORDER_METHODS, curvature_diagnostics, descend_nonconvex,
    finite_difference_gradient, gauss_newton_exponential, make_ridge_data,
    newton_quadratic, noisy_quadratic, nonconvex_stationary_points,
    optimize_ridge, quadratic_gradient_descent, ridge_gradient, ridge_loss,
    ridge_optimum, train_xor_adam, verify_gauss_newton,
    xor_loss_and_backprop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def torch_xor_gradient(parameters: np.ndarray, inputs: np.ndarray, labels: np.ndarray,
                       regularization: float) -> np.ndarray:
    tensor = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
    x = torch.tensor(inputs, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    w1 = tensor[:6].reshape(3, 2)
    b1 = tensor[6:9]
    w2 = tensor[9:12]
    b2 = tensor[12]
    hidden = torch.tanh(x @ w1.T + b1)
    logits = hidden @ w2 + b2
    loss = torch.mean(torch.logaddexp(torch.zeros_like(logits), logits) - y * logits)
    loss = loss + 0.5 * regularization * (torch.sum(w1 * w1) + torch.sum(w2 * w2))
    loss.backward()
    return tensor.grad.detach().numpy()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
               "--results-root", str(args.results_root.resolve())]
    try:
        raw_bytes = config_path.read_bytes()
        config = BenchmarkConfig.from_mapping(json.loads(raw_bytes))
        settings = config.solver_settings
        features = int(settings["features"])
        epochs = int(settings["epochs"])
        regularization = float(settings["regularization"])
        batch_size = int(settings["batch_size"])
        rates = settings["learning_rates"]
        if tuple(rates) != FIRST_ORDER_METHODS:
            raise ValueError(f"learning_rates keys must be ordered as {FIRST_ORDER_METHODS}")
        if features <= 0 or epochs <= 0 or regularization <= 0 or batch_size <= 0:
            raise ValueError("invalid Track A settings")
    except Exception as exc:
        raw_bytes = config_path.read_bytes() if config_path.is_file() else b""
        directory = record_invalid_attempt(args.results_root.resolve(), config_path,
                                           raw_bytes, exc, command)
        print(f"INVALID: {exc}\nEvidence: {directory}", file=sys.stderr)
        return 2

    recorder = RunRecorder(
        args.results_root.resolve(), config_path, config,
        source_files=[Path(__file__).resolve(), ROOT / "src/optimization_lab/continuous.py",
                      ROOT / "src/optimization_lab/benchmarking.py"],
        command=command,
    )
    started_total = time.perf_counter()
    try:
        setup_started = time.perf_counter()
        matrix, targets, true_parameters = make_ridge_data(
            20261027, config.input_size, features, observation_noise=0.2
        )
        optimum = ridge_optimum(matrix, targets, regularization)
        optimum_loss = ridge_loss(optimum, matrix, targets, regularization)
        hessian = matrix.T @ matrix / len(matrix) + regularization * np.eye(features)
        smoothness = float(np.linalg.eigvalsh(hessian).max())
        strong_convexity = float(np.linalg.eigvalsh(hessian).min())
        ridge_data_path = recorder.directory / "ridge_data.npz"
        np.savez(ridge_data_path, matrix=matrix, targets=targets,
                 true_parameters=true_parameters, optimum=optimum)
        initial_by_seed = {
            seed: np.random.default_rng(seed).normal(scale=1.5, size=features)
            for seed in config.seeds
        }
        tuning_initial = np.random.default_rng(999).normal(scale=1.5, size=features)
        tuning_epochs = min(20, epochs)
        optimizer_tuning = {}
        chosen_rates = {}
        for method in FIRST_ORDER_METHODS:
            center = float(rates[method])
            candidates = sorted({center / 4, center, center * 4})
            evaluations = []
            for candidate in candidates:
                result = optimize_ridge(
                    method, matrix, targets, regularization, tuning_initial,
                    tuning_epochs, candidate, batch_size, 999,
                )
                evaluations.append({
                    "learning_rate": candidate,
                    "final_loss": result.history[-1]["loss"],
                    "objective_gap": result.history[-1]["objective_gap"],
                    "example_gradient_evaluations": result.history[-1]["example_gradient_evaluations"],
                })
            selected = min(evaluations, key=lambda item: (item["final_loss"], item["learning_rate"]))
            chosen_rates[method] = selected["learning_rate"]
            optimizer_tuning[method] = {
                "tuning_seed": 999, "epochs": tuning_epochs,
                "candidates": evaluations, "selected_learning_rate": selected["learning_rate"],
            }
        for seed in config.seeds:
            initial = initial_by_seed[seed]
            for _ in range(config.warmup_repetitions):
                for method in FIRST_ORDER_METHODS:
                    optimize_ridge(method, matrix, targets, regularization, initial,
                                   1, float(chosen_rates[method]), batch_size, seed)
        setup_seconds = time.perf_counter() - setup_started
        trajectories: dict[str, object] = {
            "dataset_seed": 20261027,
            "initial_parameters_by_seed": {str(k): v.tolist() for k, v in initial_by_seed.items()},
            "runs": {},
        }
        deadline = started_total + config.timeout_seconds
        ridge_records = []
        for seed in config.seeds:
            trajectories["runs"][str(seed)] = {}
            initial = initial_by_seed[seed]
            initial_loss = ridge_loss(initial, matrix, targets, regularization)
            for method in FIRST_ORDER_METHODS:
                if time.perf_counter() > deadline:
                    raise TimeoutError(f"Track A exceeded {config.timeout_seconds} seconds")
                run_started = time.perf_counter()
                result = optimize_ridge(
                    method, matrix, targets, regularization, initial, epochs,
                    float(chosen_rates[method]), batch_size, seed,
                )
                elapsed = time.perf_counter() - run_started
                final = result.history[-1]
                if final["loss"] >= initial_loss:
                    raise AssertionError(f"{method} did not reduce loss for seed {seed}")
                if final["example_gradient_evaluations"] != config.input_size * epochs:
                    raise AssertionError("unequal example-gradient evaluation budget")
                record = {
                    "category": "ridge_optimizer", "seed": seed, "method": method,
                    "initial_loss": initial_loss, "final_loss": final["loss"],
                    "objective_gap": final["objective_gap"],
                    "gradient_norm": final["gradient_norm"],
                    "example_gradient_evaluations": final["example_gradient_evaluations"],
                    "parameter_updates": final["parameter_updates"],
                    "execution_seconds": elapsed,
                    "final_parameters": result.parameters.tolist(),
                    "learning_rate": float(chosen_rates[method]),
                }
                recorder.add_record(record)
                ridge_records.append(record)
                trajectories["runs"][str(seed)][method] = list(result.history)

        # Step-size sensitivity and convexity classes use known diagonal quadratics.
        step_sizes = [0.01, 0.1, 0.19, 0.21]
        step_sweep = {
            str(rate): quadratic_gradient_descent([1.0, 10.0], [2.0, 3.0], rate, 60)
            for rate in step_sizes
        }
        convex = quadratic_gradient_descent([1.0, 0.0], [2.0, 3.0], 0.5, 40)
        strongly_convex = quadratic_gradient_descent([1.0, 10.0], [2.0, 3.0], 0.1, 60)
        continuous_discrete = {
            "objective": "(x-2.3)^2",
            "continuous_optimum": {"x": 2.3, "objective": 0.0},
            "integer_candidates": [
                {"x": value, "objective": (value - 2.3) ** 2} for value in range(0, 5)
            ],
            "discrete_optimum": {"x": 2, "objective": (2 - 2.3) ** 2},
        }
        stationary = nonconvex_stationary_points()
        basin_runs = [descend_nonconvex(start) for start in (-1.5, -0.2, 0.2, 1.5)]
        saddle = {
            "objective": "x^2-y^2", "stationary_point": [0.0, 0.0],
            "gradient_at_stationary": [0.0, 0.0], "hessian_eigenvalues": [-2.0, 2.0],
            "exact_start_after_20_steps": [0.0, 0.0],
            "perturbed_y_after_20_steps_lr_0.1": 1e-6 * (1.2 ** 20),
        }
        foundations = {
            "continuous_vs_discrete": continuous_discrete,
            "convex_not_strongly_convex": {
                "hessian_eigenvalues": [0.0, 1.0], "trajectory": convex,
                "observation": "objective converges but flat-coordinate distance need not",
            },
            "strongly_convex": {
                "hessian_eigenvalues": [1.0, 10.0], "trajectory": strongly_convex,
                "observation": "unique minimizer and geometric convergence for stable step size",
            },
            "step_size_sensitivity": {
                "smoothness_L": 10.0, "stability_condition": "0 < learning_rate < 2/L",
                "trajectories": step_sweep,
            },
            "nonconvex_local_global": {"stationary_points": stationary, "basin_runs": basin_runs},
            "saddle": saddle,
        }

        # Newton and Gauss-Newton use objectives appropriate to their assumptions.
        newton = newton_quadratic(
            np.diag([1.0, 4.0, 9.0]), np.array([1.0, -2.0, 3.0]),
            np.array([3.0, 3.0, 3.0]), iterations=3,
        )
        curve_x = np.linspace(0.0, 1.0, 40)
        curve_rng = np.random.default_rng(91)
        curve_y = 1.5 * np.exp(0.7 * curve_x) + curve_rng.normal(scale=0.01, size=len(curve_x))
        curve_initial = np.array([1.0, 0.2])
        gauss_newton = gauss_newton_exponential(curve_x, curve_y, curve_initial)
        gauss_check = verify_gauss_newton(curve_x, curve_y, curve_initial, gauss_newton)
        second_order = {
            "newton_quadratic": {
                "parameters": newton.parameters.tolist(), "history": list(newton.history),
                "assumption": "constant positive-definite Hessian; exact solve",
            },
            "gauss_newton_nonlinear_least_squares": {
                "x": curve_x.tolist(), "y": curve_y.tolist(),
                "initial": curve_initial.tolist(), "parameters": gauss_newton.parameters.tolist(),
                "history": list(gauss_newton.history), "scipy_verification": gauss_check,
                "assumption": "J^T J approximates residual Hessian; local method",
            },
        }

        # Manual backprop is verified independently, then used by Adam to train XOR.
        xor_inputs = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        xor_labels = np.array([0., 1., 1., 0.])
        xor_initial = np.random.default_rng(5).normal(scale=0.3, size=13)
        xor_loss, manual_gradient, _ = xor_loss_and_backprop(xor_initial, xor_inputs, xor_labels)
        finite_gradient = finite_difference_gradient(
            lambda p: xor_loss_and_backprop(p, xor_inputs, xor_labels)[0], xor_initial
        )
        torch_gradient = torch_xor_gradient(xor_initial, xor_inputs, xor_labels, 1e-3)
        xor_trained = train_xor_adam(xor_initial)
        _, _, final_probabilities = xor_loss_and_backprop(
            xor_trained.parameters, xor_inputs, xor_labels
        )
        backprop = {
            "statement": "backpropagation computes gradients; Adam applies those gradients",
            "parameter_count": 13, "loss": xor_loss,
            "manual_vs_finite_relative_error": float(
                np.linalg.norm(manual_gradient - finite_gradient) / np.linalg.norm(finite_gradient)
            ),
            "manual_vs_torch_relative_error": float(
                np.linalg.norm(manual_gradient - torch_gradient) / np.linalg.norm(torch_gradient)
            ),
            "initial_parameters": xor_initial.tolist(),
            "final_parameters": xor_trained.parameters.tolist(),
            "training_history": list(xor_trained.history),
            "final_probabilities": final_probabilities.tolist(),
            "final_accuracy": float(np.mean((final_probabilities >= 0.5) == xor_labels)),
        }

        # Logistic curvature: exact Hessian, Fisher variants, diagonal/low-rank and HVP.
        curvature_rng = np.random.default_rng(77)
        curvature_x = curvature_rng.normal(size=(80, 5))
        curvature_true = curvature_rng.normal(size=5)
        curvature_probabilities = 1 / (1 + np.exp(-(curvature_x @ curvature_true)))
        curvature_y = (curvature_rng.random(80) < curvature_probabilities).astype(float)
        curvature_point = np.linspace(-0.2, 0.2, 5)
        direction = np.array([1.0, -0.5, 0.25, 0.75, -1.0])
        curvature = curvature_diagnostics(
            curvature_point, curvature_x, curvature_y, 0.05, direction, rank=2
        )
        curvature["parameters"] = curvature_point.tolist()
        curvature["matrix"] = curvature_x.tolist()
        curvature["labels"] = curvature_y.tolist()
        curvature["regularization"] = 0.05
        curvature["direction"] = direction.tolist()

        noise_records = []
        for seed in config.seeds:
            for strategy in ("constant_step", "decaying_step"):
                noise_record = noisy_quadratic(seed, strategy)
                noise_record["category"] = "noisy_objective"
                recorder.add_record(noise_record)
                noise_records.append(noise_record)

        # Persist raw analyses before plotting.
        trajectories_path = recorder.directory / "ridge_trajectories.json"
        foundations_path = recorder.directory / "foundations.json"
        second_order_path = recorder.directory / "second_order.json"
        backprop_path = recorder.directory / "backpropagation.json"
        curvature_path = recorder.directory / "curvature.json"
        noise_path = recorder.directory / "noisy_objective.json"
        tuning_path = recorder.directory / "optimizer_tuning.json"
        write_json(trajectories_path, trajectories)
        write_json(foundations_path, foundations)
        write_json(second_order_path, second_order)
        write_json(backprop_path, backprop)
        write_json(curvature_path, curvature)
        write_json(noise_path, noise_records)
        write_json(tuning_path, {
            "protocol": "three candidate rates per method on fixed seed 999 before measured seeds",
            "initial_parameters": tuning_initial.tolist(),
            "methods": optimizer_tuning,
        })

        # Overview plot.
        overview = recorder.directory / "continuous_overview.png"
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        epochs_axis = np.arange(epochs + 1)
        for method in FIRST_ORDER_METHODS:
            values = np.array([
                [entry["objective_gap"] for entry in trajectories["runs"][str(seed)][method]]
                for seed in config.seeds
            ])
            axes[0, 0].plot(epochs_axis, np.maximum(values.mean(axis=0), 1e-16), label=method)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("Ridge objective gap (mean across seeds)")
        axes[0, 0].set_xlabel("Epoch / full data exposure")
        axes[0, 0].set_ylabel("Objective gap (log scale)")
        axes[0, 0].legend(fontsize=7, ncol=2)
        gap_summary = numeric_summary(ridge_records, "method", "objective_gap")
        axes[0, 1].bar(FIRST_ORDER_METHODS,
                       [max(gap_summary[method]["mean"], 1e-16)
                        for method in FIRST_ORDER_METHODS])
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_title("Final ridge gap (exact zero plotted at 1e-16)")
        axes[0, 1].tick_params(axis="x", rotation=30)
        for rate, trajectory in step_sweep.items():
            axes[1, 0].plot([entry["iteration"] for entry in trajectory],
                            [max(entry["objective"], 1e-16) for entry in trajectory], label=rate)
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_title("Step-size sensitivity; L=10")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Objective (log scale)")
        axes[1, 0].legend(title="learning rate")
        noise_summary = numeric_summary(noise_records, "strategy", "absolute_error")
        strategies = ["constant_step", "decaying_step"]
        axes[1, 1].bar(strategies, [noise_summary[s]["mean"] for s in strategies],
                       yerr=[noise_summary[s]["sample_std"] for s in strategies], capsize=5)
        axes[1, 1].set_title("Noisy gradient: final |x-3|")
        axes[1, 1].set_ylabel("Mean ± sample SD over seeds")
        for axis in axes.ravel():
            axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(overview, dpi=140)
        plt.close(fig)

        landscape_plot = recorder.directory / "nonconvex_landscape.png"
        grid = np.linspace(-1.25, 1.25, 500)
        values = grid ** 4 - grid ** 2 + 0.1 * grid
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(grid, values, label=r"$x^4-x^2+0.1x$")
        for point in stationary:
            axis.scatter(point["x"], point["objective"], s=55,
                         label=point["classification"].replace("_", " "))
        axis.set_xlabel("x")
        axis.set_ylabel("Objective")
        axis.set_title("Non-convex landscape: local/global minima and maximum")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(landscape_plot, dpi=140)
        plt.close(fig)

        recorder.metadata["timing_policy"] = {
            "setup_included_in_execution": False, "setup_seconds": setup_seconds,
            "ridge_budget": "equal examples processed per epoch and equal epoch count",
            "optimizer_tuning_included_in_setup": True,
            "optimizer_tuning_seed": 999,
            "measured_seeds": list(config.seeds), "clock": "time.perf_counter",
        }
        recorder.metadata["ridge_problem"] = {
            "parameters": features, "samples": config.input_size,
            "regularization": regularization, "smoothness_L": smoothness,
            "strong_convexity_mu": strong_convexity, "condition_number": smoothness / strong_convexity,
            "known_optimum_loss": optimum_loss,
        }
        (recorder.directory / "metadata.json").write_bytes(canonical_json_bytes(recorder.metadata))
        timing_summary = numeric_summary(ridge_records, "method", "execution_seconds")
        gap_summary = numeric_summary(ridge_records, "method", "objective_gap")
        summary = {
            "status": "completed", "scope": "Track A continuous optimization",
            "ridge_records": len(ridge_records), "noise_records": len(noise_records),
            "seeds": list(config.seeds), "epochs": epochs,
            "equal_example_gradient_evaluations": config.input_size * epochs,
            "ridge_timing_by_method": timing_summary,
            "ridge_final_gap_by_method": gap_summary,
            "selected_learning_rates": chosen_rates,
            "newton_final_distance": newton.history[-1]["distance_to_optimum"],
            "gauss_newton_scipy_parameter_distance": gauss_check["parameter_distance"],
            "backprop_finite_difference_relative_error": backprop["manual_vs_finite_relative_error"],
            "backprop_torch_relative_error": backprop["manual_vs_torch_relative_error"],
            "xor_final_accuracy": backprop["final_accuracy"],
            "hvp_relative_error": curvature["hvp_relative_error"],
            "expected_fisher_relative_error": curvature["expected_fisher_relative_error"],
            "noisy_objective_by_strategy": noise_summary,
        }
        recorder.complete(summary, {
            "ridge_data": ridge_data_path, "ridge_trajectories": trajectories_path,
            "foundations": foundations_path, "second_order": second_order_path,
            "backpropagation": backprop_path, "curvature": curvature_path,
            "noisy_objective": noise_path, "overview_plot": overview,
            "nonconvex_plot": landscape_plot, "optimizer_tuning": tuning_path,
        })
        print(recorder.directory)
        return 0
    except Exception as exc:
        recorder.fail(exc)
        print(f"FAILED: {exc}\nEvidence: {recorder.directory}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

