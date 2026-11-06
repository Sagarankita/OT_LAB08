#!/usr/bin/env python3
"""Independently verify Track C quantum-optimization artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
import pennylane as qml

sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.quantum_optimization import (  # noqa: E402
    apply_qaoa,
    apply_vqe_ansatz,
    cost_hamiltonian,
    finite_difference_derivative,
    load_qubo,
    parameter_shift_derivative,
    qubo_energy,
    verify_hamiltonian_basis,
    vqe_hamiltonian,
)


def load(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run_dir = args.run_directory.resolve()
    config = load(run_dir / "config.json")
    raw = load(run_dir / "raw_results.json")
    summary = load(run_dir / "summary.json")
    source_path = ROOT / "data" / "track_b_qubo.json"
    payload = load_qubo(source_path)
    qubo = payload["qubo"]

    basis = verify_hamiltonian_basis(qubo)
    enumerated = [
        (list(bits), qubo_energy(bits, qubo)) for bits in product((0, 1), repeat=qubo["n"])
    ]
    minimum = min(energy for _, energy in enumerated)
    minimizers = [bits for bits, energy in enumerated if energy == minimum]

    best_qaoa = raw["qaoa"]["best_run"]
    qaoa_device = qml.device("default.qubit", wires=qubo["n"])
    hamiltonian = cost_hamiltonian(qubo)

    @qml.qnode(qaoa_device)
    def qaoa_energy(parameters):
        apply_qaoa(parameters, qubo)
        return qml.expval(hamiltonian)

    reproduced_qaoa = float(qaoa_energy(np.array(best_qaoa["parameters"])))
    all_samples = raw["qaoa"]["finite_shots"] + raw["qaoa"]["noisy"]
    sample_counts_valid = all(
        sum(state["count"] for state in item["states"]) == item["shots"] for item in all_samples
    )
    probabilities_valid = all(
        math.isclose(sum(state["probability"] for state in item["states"]), 1.0, abs_tol=1e-12)
        for item in all_samples
    )
    optimum_sampled = all(
        item["best_feasible"] is not None and item["best_feasible"]["bits"] == minimizers[0]
        for item in all_samples
    )
    noise_labels = all("simulated" in item["noise"] for item in raw["qaoa"]["noisy"])

    vqe_matrix = qml.matrix(vqe_hamiltonian(), wire_order=[0, 1])
    exact_vqe = float(np.linalg.eigvalsh(vqe_matrix)[0])
    vqe_device = qml.device("default.qubit", wires=2)

    @qml.qnode(vqe_device)
    def vqe_energy(parameters):
        apply_vqe_ansatz(parameters)
        return qml.expval(vqe_hamiltonian())

    vqe_reproduced = all(
        math.isclose(float(vqe_energy(np.array(item["parameters"]))), item["energy"], abs_tol=1e-9)
        for item in raw["vqe"]["optimizers"]
    )
    variational_bounds = all(item["energy"] >= exact_vqe - 1e-9 for item in raw["vqe"]["optimizers"])
    qng_record = next(item for item in raw["vqe"]["optimizers"] if item["method"] == "quantum_natural_gradient")
    spsa_record = next(item for item in raw["vqe"]["optimizers"] if item["method"] == "spsa")

    derivative_device = qml.device("default.qubit", wires=1)

    @qml.qnode(derivative_device)
    def derivative_circuit(parameters):
        qml.RY(parameters[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    derivative_parameters = np.array([0.41])
    shifted = parameter_shift_derivative(derivative_circuit, derivative_parameters, 0)
    finite = finite_difference_derivative(derivative_circuit, derivative_parameters, 0)

    variance_records = raw["gradient_variance"]["records"]
    variance_reproduced = all(
        math.isclose(item["gradient_variance"], float(np.var(item["gradients"], ddof=1)), rel_tol=1e-12)
        for item in variance_records
    )
    depth_five = sorted(
        (item for item in variance_records if item["depth"] == max(config["gradient_depths"])),
        key=lambda item: item["qubits"],
    )
    observed_decrease = all(
        depth_five[index + 1]["gradient_variance"] < depth_five[index]["gradient_variance"]
        for index in range(len(depth_five) - 1)
    )

    checks = {
        "final_configuration": config["run_kind"] == "final",
        "source_hash_matches": hashlib.sha256(source_path.read_bytes()).hexdigest() == raw["qubo_source"]["sha256"],
        "track_b_instruction_preserved": "do not regenerate or rescale" in raw["qubo_source"]["instruction"],
        "all_32_basis_states_match": basis["passed"] and len(basis["states"]) == 32,
        "classical_optimum_reproduced": minimum == -18.0 and minimizers == [[1, 1, 0, 0, 0]],
        "qaoa_energy_reproduced": math.isclose(reproduced_qaoa, best_qaoa["final_expected_energy"], abs_tol=1e-9),
        "qaoa_variational_bound": best_qaoa["final_expected_energy"] >= minimum - 1e-9,
        "three_qaoa_seeds": len(raw["qaoa"]["optimization_runs"]) == 3,
        "shot_budgets_present": {item["shots"] for item in raw["qaoa"]["finite_shots"]} == {100, 1000, 5000},
        "sample_counts_equal_shots": sample_counts_valid,
        "sample_probabilities_sum_one": probabilities_valid,
        "optimal_feasible_sample_seen": optimum_sampled,
        "noise_explicitly_simulated": noise_labels,
        "noise_degrades_mean_energy": summary["qaoa"]["noisy_mean_energy"] > summary["qaoa"]["shot_summary"][-1]["mean_energy"],
        "shot_variability_decreases": summary["qaoa"]["shot_summary"][-1]["energy_stdev"] < summary["qaoa"]["shot_summary"][0]["energy_stdev"],
        "vqe_exact_diagonalization_reproduced": math.isclose(exact_vqe, raw["vqe"]["exact_ground_energy"], abs_tol=1e-12),
        "vqe_final_energies_reproduced": vqe_reproduced,
        "vqe_variational_bounds": variational_bounds,
        "all_vqe_methods_near_ground": all(item["error_to_exact"] < 0.001 for item in raw["vqe"]["optimizers"]),
        "parameter_shift_matches_finite_difference": abs(shifted - finite) < 1e-7 and raw["vqe"]["derivative_check"]["absolute_error"] < 1e-7,
        "spsa_two_point_accounting": spsa_record["expected_two_point_calls"] == 2 * config["vqe_spsa_steps"] + 1 and spsa_record["circuit_executions"] >= spsa_record["expected_two_point_calls"],
        "qng_metric_recorded": "Fubini-Study" in qng_record["metric"] and len(qng_record["metric_condition_numbers"]) == config["vqe_qng_steps"],
        "qng_metric_finite": all(math.isfinite(value) and value > 0 for value in qng_record["metric_condition_numbers"]),
        "qnn_three_seeds": len(raw["qnn"]["runs"]) == 3,
        "qnn_circuit_cost_recorded": all(item["circuit_executions"] > 0 and item["parameter_count"] == 4 for item in raw["qnn"]["runs"]),
        "qnn_accuracy_nontrivial": summary["qnn"]["mean_test_accuracy"] >= 0.75,
        "gradient_grid_complete": len(variance_records) == 9,
        "gradient_variances_reproduced": variance_reproduced,
        "parameter_shift_cost_for_variance": all(item["circuit_executions"] == 2 * item["seeds"] for item in variance_records),
        "small_variance_trend_observed": observed_decrease,
        "barren_plateau_claim_cautious": "not proof" in raw["gradient_variance"]["interpretation"],
        "simulator_only_disclosure": "no real quantum hardware" in summary["environment"]["execution"],
        "plot_present": (run_dir / "quantum_optimization_comparison.png").stat().st_size > 0,
        "status_passed": load(run_dir / "status.json")["passed"] is True,
    }
    result = {"passed": all(checks.values()), "check_count": len(checks), "checks": checks}
    (run_dir / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
