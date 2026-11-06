"""Independently audit a completed Track A result."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.benchmarking import load_completed_run, sha256_file
from optimization_lab.continuous import (
    FIRST_ORDER_METHODS, curvature_diagnostics, finite_difference_gradient,
    newton_quadratic, optimize_ridge, ridge_gradient, ridge_loss, ridge_optimum,
    verify_gauss_newton, gauss_newton_exponential, xor_loss_and_backprop,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    loaded = load_completed_run(run)
    config, metadata, records, summary = (
        loaded["config"], loaded["metadata"], loaded["records"], loaded["summary"]
    )
    ridge_records = [record for record in records if record["category"] == "ridge_optimizer"]
    noise_records = [record for record in records if record["category"] == "noisy_objective"]
    data = np.load(run / "ridge_data.npz")
    matrix, targets, saved_optimum = data["matrix"], data["targets"], data["optimum"]
    regularization = float(config["solver_settings"]["regularization"])
    optimum = ridge_optimum(matrix, targets, regularization)
    optimum_loss = ridge_loss(optimum, matrix, targets, regularization)
    recomputed_records = all(
        math.isclose(record["final_loss"], ridge_loss(
            np.asarray(record["final_parameters"]), matrix, targets, regularization
        ), rel_tol=1e-11, abs_tol=1e-12)
        and math.isclose(record["objective_gap"], max(0.0, record["final_loss"] - optimum_loss),
                         rel_tol=1e-9, abs_tol=1e-12)
        for record in ridge_records
    )
    paired = all(
        {record["method"] for record in ridge_records if record["seed"] == seed}
        == set(FIRST_ORDER_METHODS) for seed in config["seeds"]
    )
    point = np.linspace(-0.3, 0.3, matrix.shape[1])
    analytic = ridge_gradient(point, matrix, targets, regularization)
    finite = finite_difference_gradient(
        lambda p: ridge_loss(p, matrix, targets, regularization), point
    )
    gradient_relative_error = float(np.linalg.norm(analytic - finite) / np.linalg.norm(finite))
    tuning = json.loads((run / "optimizer_tuning.json").read_text())
    tuning_initial = np.asarray(tuning["initial_parameters"])
    tuning_reproduced = True
    selected_rates = {}
    for method in FIRST_ORDER_METHODS:
        saved_method = tuning["methods"][method]
        recomputed = []
        for candidate in saved_method["candidates"]:
            result = optimize_ridge(
                method, matrix, targets, regularization, tuning_initial,
                int(saved_method["epochs"]), float(candidate["learning_rate"]),
                int(config["solver_settings"]["batch_size"]), int(saved_method["tuning_seed"]),
            )
            recomputed.append((candidate["learning_rate"], result.history[-1]["loss"]))
            tuning_reproduced &= math.isclose(
                result.history[-1]["loss"], candidate["final_loss"], rel_tol=1e-12, abs_tol=1e-12
            )
        selected = min(recomputed, key=lambda item: (item[1], item[0]))[0]
        selected_rates[method] = selected
        tuning_reproduced &= selected == saved_method["selected_learning_rate"]
    measured_rates_match = all(
        record["learning_rate"] == selected_rates[record["method"]] for record in ridge_records
    )

    foundations = json.loads((run / "foundations.json").read_text())
    sweep = foundations["step_size_sensitivity"]["trajectories"]
    stable_step_converges = sweep["0.1"][-1]["objective"] < sweep["0.1"][0]["objective"]
    unstable_step_diverges = sweep["0.21"][-1]["objective"] > sweep["0.21"][0]["objective"]
    convex_flat_coordinate = foundations["convex_not_strongly_convex"]["trajectory"][-1]["point"][1]
    stationary_classes = {
        item["classification"] for item in foundations["nonconvex_local_global"]["stationary_points"]
    }

    second_order = json.loads((run / "second_order.json").read_text())
    newton = newton_quadratic(
        np.diag([1.0, 4.0, 9.0]), np.array([1.0, -2.0, 3.0]),
        np.array([3.0, 3.0, 3.0]), iterations=3,
    )
    gauss_saved = second_order["gauss_newton_nonlinear_least_squares"]
    curve_x, curve_y = np.asarray(gauss_saved["x"]), np.asarray(gauss_saved["y"])
    curve_initial = np.asarray(gauss_saved["initial"])
    gauss = gauss_newton_exponential(curve_x, curve_y, curve_initial)
    gauss_check = verify_gauss_newton(curve_x, curve_y, curve_initial, gauss)

    backprop = json.loads((run / "backpropagation.json").read_text())
    xor_inputs = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    xor_labels = np.array([0., 1., 1., 0.])
    initial = np.asarray(backprop["initial_parameters"])
    _, manual, _ = xor_loss_and_backprop(initial, xor_inputs, xor_labels)
    numeric = finite_difference_gradient(
        lambda p: xor_loss_and_backprop(p, xor_inputs, xor_labels)[0], initial
    )
    backprop_relative_error = float(np.linalg.norm(manual - numeric) / np.linalg.norm(numeric))
    _, _, final_probabilities = xor_loss_and_backprop(
        np.asarray(backprop["final_parameters"]), xor_inputs, xor_labels
    )
    xor_accuracy = float(np.mean((final_probabilities >= 0.5) == xor_labels))

    curvature_saved = json.loads((run / "curvature.json").read_text())
    curvature = curvature_diagnostics(
        np.asarray(curvature_saved["parameters"]), np.asarray(curvature_saved["matrix"]),
        np.asarray(curvature_saved["labels"]), float(curvature_saved["regularization"]),
        np.asarray(curvature_saved["direction"]), rank=int(curvature_saved["rank"]),
    )
    status = json.loads((run / "status.json").read_text())
    artifact_hashes = all(
        sha256_file(run / artifact["path"]) == artifact["sha256"]
        for artifact in status["artifacts"].values()
    )
    expected_ridge_records = len(config["seeds"]) * len(FIRST_ORDER_METHODS)
    expected_noise_records = len(config["seeds"]) * 2
    checks = {
        "final_configuration": config["run_kind"] == "final",
        "five_stochastic_seeds": len(config["seeds"]) >= 5,
        "ridge_record_count": len(ridge_records) == expected_ridge_records == summary["ridge_records"],
        "noisy_record_count": len(noise_records) == expected_noise_records == summary["noise_records"],
        "all_optimizers_paired_per_seed": paired,
        "equal_example_gradient_budget": len({r["example_gradient_evaluations"] for r in ridge_records}) == 1,
        "every_optimizer_reduced_loss": all(r["final_loss"] < r["initial_loss"] for r in ridge_records),
        "known_optimum_recomputed": np.allclose(optimum, saved_optimum, atol=1e-12),
        "saved_losses_and_gaps_recomputed": recomputed_records,
        "ridge_gradient_matches_finite_difference": gradient_relative_error < 1e-7,
        "optimizer_tuning_reproduced": tuning_reproduced,
        "measured_runs_use_selected_rates": measured_rates_match,
        "convex_flat_direction_demonstrated": math.isclose(convex_flat_coordinate, 3.0),
        "stable_step_converges": stable_step_converges,
        "above_limit_step_diverges": unstable_step_diverges,
        "local_global_stationary_classes_present": stationary_classes == {
            "global_minimum", "nonglobal_local_minimum", "local_maximum"
        },
        "saddle_has_mixed_curvature": foundations["saddle"]["hessian_eigenvalues"] == [-2.0, 2.0],
        "newton_reaches_exact_quadratic_optimum": newton.history[1]["distance_to_optimum"] < 1e-12,
        "gauss_newton_matches_scipy": gauss_check["scipy_success"] and
                                      gauss_check["parameter_distance"] < 1e-6,
        "manual_backprop_matches_finite_difference": backprop_relative_error < 1e-5,
        "manual_backprop_matches_torch_saved": backprop["manual_vs_torch_relative_error"] < 1e-10,
        "xor_trains_to_full_accuracy": xor_accuracy == 1.0 == backprop["final_accuracy"],
        "hvp_matches_gradient_finite_difference": curvature["hvp_relative_error"] < 1e-7,
        "expected_fisher_identity": curvature["expected_fisher_relative_error"] < 1e-12,
        "curvature_approximations_finite": all(math.isfinite(curvature[key]) for key in (
            "empirical_fisher_relative_error", "diagonal_relative_error", "low_rank_relative_error"
        )),
        "noisy_objective_values_recomputed": all(math.isclose(
            record["final_objective"], 0.5 * (record["final_x"] - 3.0) ** 2,
            abs_tol=1e-12,
        ) for record in noise_records),
        "timing_policy_and_git_provenance": metadata["timing_policy"]["setup_included_in_execution"] is False
                                              and metadata["provenance"]["git_revision"]
                                              == "pending user Git work",
        "all_declared_artifact_hashes_match": artifact_hashes,
    }
    report = {
        "checks": checks, "passed": all(checks.values()),
        "ridge_records": len(ridge_records), "noise_records": len(noise_records),
        "ridge_gradient_relative_error": gradient_relative_error,
        "backprop_relative_error": backprop_relative_error,
    }
    (run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
