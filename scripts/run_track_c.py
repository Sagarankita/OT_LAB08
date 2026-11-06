#!/usr/bin/env python3
"""Run Track C quantum-optimization experiments on local simulators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.quantum_optimization import (  # noqa: E402
    analyze_counts,
    apply_qaoa,
    apply_vqe_ansatz,
    cost_hamiltonian,
    finite_difference_derivative,
    load_qubo,
    parameter_shift_derivative,
    verify_hamiltonian_basis,
    vqe_hamiltonian,
)


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def executions(tracker: qml.Tracker) -> int:
    return int(tracker.totals.get("executions", 0))


def qaoa_experiment(payload: dict, config: dict) -> dict:
    qubo = payload["qubo"]
    hamiltonian = cost_hamiltonian(qubo)
    runs = []
    for seed in config["seeds"]:
        device = qml.device("default.qubit", wires=qubo["n"])

        @qml.qnode(device, interface="autograd", diff_method="backprop")
        def energy(parameters):
            apply_qaoa(parameters, qubo)
            return qml.expval(hamiltonian)

        rng = np.random.default_rng(seed)
        parameters = pnp.array(
            rng.uniform(-0.1, 0.1, size=(config["qaoa_depth"], 2)), requires_grad=True
        )
        optimizer = qml.AdamOptimizer(config["qaoa_learning_rate"])
        history = []
        started = time.perf_counter()
        with qml.Tracker(device) as tracker:
            for step in range(config["qaoa_steps"]):
                parameters, old_energy = optimizer.step_and_cost(energy, parameters)
                history.append({"step": step + 1, "energy": float(old_energy)})
            final_energy = float(energy(parameters))
        runs.append(
            {
                "seed": seed,
                "depth": config["qaoa_depth"],
                "steps": config["qaoa_steps"],
                "parameters": np.asarray(parameters).tolist(),
                "final_expected_energy": final_energy,
                "runtime_seconds": time.perf_counter() - started,
                "circuit_executions": executions(tracker),
                "gradient_method": "analytic statevector backpropagation",
                "history": history,
            }
        )
    best = min(runs, key=lambda item: item["final_expected_energy"])
    best_parameters = np.array(best["parameters"])
    sampling = []
    for shots in config["shot_budgets"]:
        for seed in config["seeds"]:
            device = qml.device("default.qubit", wires=qubo["n"], shots=shots, seed=seed)

            @qml.qnode(device)
            def sample_circuit():
                apply_qaoa(best_parameters, qubo)
                return qml.counts(wires=range(qubo["n"]))

            with qml.Tracker(device) as tracker:
                counts = sample_circuit()
            analysis = analyze_counts(counts, payload)
            analysis.update(
                {
                    "seed": seed,
                    "noise": "none; ideal finite-shot simulator",
                    "circuit_executions": executions(tracker),
                }
            )
            sampling.append(analysis)

    noise_shots = max(config["shot_budgets"])
    noisy_sampling = []
    for seed in config["seeds"]:
        device = qml.device("default.mixed", wires=qubo["n"], shots=noise_shots, seed=seed)

        @qml.qnode(device)
        def noisy_circuit():
            apply_qaoa(best_parameters, qubo, config["noise_probability"])
            return qml.counts(wires=range(qubo["n"]))

        with qml.Tracker(device) as tracker:
            counts = noisy_circuit()
        analysis = analyze_counts(counts, payload)
        analysis.update(
            {
                "seed": seed,
                "noise": f"simulated layerwise single-qubit depolarizing channel p={config['noise_probability']}",
                "circuit_executions": executions(tracker),
            }
        )
        noisy_sampling.append(analysis)
    return {"optimization_runs": runs, "best_run": best, "finite_shots": sampling, "noisy": noisy_sampling}


def vqe_experiment(config: dict) -> dict:
    hamiltonian = vqe_hamiltonian()
    matrix = qml.matrix(hamiltonian, wire_order=[0, 1])
    exact_ground = float(np.linalg.eigvalsh(matrix)[0])
    initial_values = np.array([0.2, 0.1, -0.3, 0.2])
    results = []

    # Parameter-shift Adam.
    device = qml.device("default.qubit", wires=2)

    @qml.qnode(device, interface="autograd", diff_method="parameter-shift")
    def energy_ps(parameters):
        apply_vqe_ansatz(parameters)
        return qml.expval(hamiltonian)

    parameters = pnp.array(initial_values, requires_grad=True)
    optimizer = qml.AdamOptimizer(0.08)
    history = []
    with qml.Tracker(device) as tracker:
        for step in range(config["vqe_parameter_shift_steps"]):
            parameters, old_energy = optimizer.step_and_cost(energy_ps, parameters)
            history.append(float(old_energy))
        final_energy = float(energy_ps(parameters))
    results.append(
        {
            "method": "parameter_shift_adam",
            "iterations": config["vqe_parameter_shift_steps"],
            "energy": final_energy,
            "error_to_exact": final_energy - exact_ground,
            "parameters": np.asarray(parameters).tolist(),
            "circuit_executions": executions(tracker),
            "history": history,
        }
    )

    # SPSA: two objective circuits per stochastic gradient plus final evaluation.
    device_spsa = qml.device("default.qubit", wires=2)

    @qml.qnode(device_spsa)
    def energy_spsa(parameters):
        apply_vqe_ansatz(parameters)
        return qml.expval(hamiltonian)

    parameters_spsa = initial_values.copy()
    rng = np.random.default_rng(3)
    history_spsa = []
    with qml.Tracker(device_spsa) as tracker_spsa:
        for step in range(config["vqe_spsa_steps"]):
            a = 1.0 / (step + 10) ** 0.602
            c = 0.15 / (step + 1) ** 0.101
            direction = rng.choice([-1.0, 1.0], size=len(parameters_spsa))
            plus = float(energy_spsa(parameters_spsa + c * direction))
            minus = float(energy_spsa(parameters_spsa - c * direction))
            gradient = ((plus - minus) / (2 * c)) * direction
            parameters_spsa -= a * gradient
            history_spsa.append((plus + minus) / 2)
        final_spsa = float(energy_spsa(parameters_spsa))
    results.append(
        {
            "method": "spsa",
            "iterations": config["vqe_spsa_steps"],
            "energy": final_spsa,
            "error_to_exact": final_spsa - exact_ground,
            "parameters": parameters_spsa.tolist(),
            "circuit_executions": executions(tracker_spsa),
            "expected_two_point_calls": 2 * config["vqe_spsa_steps"] + 1,
            "history": history_spsa,
        }
    )

    # Metric-based quantum natural gradient with a block-diagonal Fubini-Study metric.
    device_qng = qml.device("default.qubit", wires=2)

    @qml.qnode(device_qng, interface="autograd", diff_method="parameter-shift")
    def energy_qng(parameters):
        apply_vqe_ansatz(parameters)
        return qml.expval(hamiltonian)

    @qml.qnode(device_qng, interface="autograd")
    def state_qng(parameters):
        apply_vqe_ansatz(parameters)
        return qml.state()

    parameters_qng = pnp.array(initial_values, requires_grad=True)
    history_qng = []
    metric_condition_numbers = []
    with qml.Tracker(device_qng) as tracker_qng:
        for _ in range(config["vqe_qng_steps"]):
            gradient = qml.grad(energy_qng)(parameters_qng)
            metric = qml.metric_tensor(state_qng, approx="block-diag")(parameters_qng)
            regularized = metric + 1e-3 * pnp.eye(len(parameters_qng))
            metric_condition_numbers.append(float(np.linalg.cond(np.asarray(regularized))))
            parameters_qng = parameters_qng - 0.08 * pnp.linalg.pinv(regularized) @ gradient
            history_qng.append(float(energy_qng(parameters_qng)))
        final_qng = float(energy_qng(parameters_qng))
    results.append(
        {
            "method": "quantum_natural_gradient",
            "metric": "block-diagonal Fubini-Study metric with 1e-3 diagonal regularization",
            "iterations": config["vqe_qng_steps"],
            "energy": final_qng,
            "error_to_exact": final_qng - exact_ground,
            "parameters": np.asarray(parameters_qng).tolist(),
            "circuit_executions": executions(tracker_qng),
            "metric_condition_numbers": metric_condition_numbers,
            "history": history_qng,
        }
    )

    derivative_device = qml.device("default.qubit", wires=2)

    @qml.qnode(derivative_device)
    def derivative_circuit(parameters):
        apply_vqe_ansatz(parameters)
        return qml.expval(qml.PauliZ(0))

    derivative_parameters = np.array([0.31, -0.2, 0.17, 0.4])
    shifted = parameter_shift_derivative(derivative_circuit, derivative_parameters, 0)
    finite = finite_difference_derivative(derivative_circuit, derivative_parameters, 0)
    derivative_check = {
        "parameter_shift": shifted,
        "finite_difference": finite,
        "absolute_error": abs(shifted - finite),
        "parameter_shift_circuit_calls": 2,
        "finite_difference_circuit_calls": 2,
    }
    return {"exact_ground_energy": exact_ground, "optimizers": results, "derivative_check": derivative_check}


def qnn_experiment(config: dict) -> dict:
    data_rng = np.random.default_rng(20261028)
    features = data_rng.uniform(-1.0, 1.0, size=(24, 2))
    labels = np.where(features[:, 0] + 0.7 * features[:, 1] > 0, 1.0, -1.0)
    train_indices = list(range(16))
    test_indices = list(range(16, 24))
    runs = []
    for seed in config["seeds"]:
        device = qml.device("default.qubit", wires=2)

        @qml.qnode(device, interface="autograd", diff_method="parameter-shift")
        def circuit(weights, inputs):
            qml.AngleEmbedding(inputs, wires=[0, 1], rotation="Y")
            for layer in range(2):
                qml.RY(weights[layer, 0], wires=0)
                qml.RY(weights[layer, 1], wires=1)
                qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        def cost(weights):
            predictions = pnp.stack(
                [circuit(weights, features[index]) for index in train_indices]
            )
            return pnp.mean((predictions - labels[train_indices]) ** 2)

        rng = np.random.default_rng(seed)
        weights = pnp.array(rng.normal(scale=0.2, size=(2, 2)), requires_grad=True)
        optimizer = qml.AdamOptimizer(0.1)
        history = []
        started = time.perf_counter()
        with qml.Tracker(device) as tracker:
            for epoch in range(config["qnn_epochs"]):
                weights, old_cost = optimizer.step_and_cost(cost, weights)
                history.append(float(old_cost))
            train_predictions = np.array([circuit(weights, features[index]) for index in train_indices])
            test_predictions = np.array([circuit(weights, features[index]) for index in test_indices])
        runs.append(
            {
                "seed": seed,
                "epochs": config["qnn_epochs"],
                "parameter_count": int(np.size(weights)),
                "train_accuracy": float(np.mean(np.where(train_predictions >= 0, 1, -1) == labels[train_indices])),
                "test_accuracy": float(np.mean(np.where(test_predictions >= 0, 1, -1) == labels[test_indices])),
                "final_cost": float(cost(weights)),
                "runtime_seconds": time.perf_counter() - started,
                "circuit_executions": executions(tracker),
                "gradient_method": "parameter-shift",
                "history": history,
            }
        )
    return {"dataset": {"train": 16, "test": 8, "rule": "x0 + 0.7*x1 > 0"}, "runs": runs}


def gradient_variance_experiment(config: dict) -> dict:
    records = []
    for qubits in config["gradient_qubits"]:
        for depth in config["gradient_depths"]:
            gradients = []
            for seed in range(config["gradient_seeds"]):
                rng = np.random.default_rng(10_000 + 100 * qubits + 10 * depth + seed)
                parameters = rng.uniform(0, 2 * np.pi, size=(depth, qubits, 2))
                device = qml.device("default.qubit", wires=qubits)

                @qml.qnode(device)
                def circuit(values):
                    for layer in range(depth):
                        for wire in range(qubits):
                            qml.RY(values[layer, wire, 0], wires=wire)
                            qml.RZ(values[layer, wire, 1], wires=wire)
                        for wire in range(qubits):
                            qml.CNOT(wires=[wire, (wire + 1) % qubits])
                    observable = qml.prod(*(qml.PauliZ(wire) for wire in range(qubits)))
                    return qml.expval(observable)

                plus = parameters.copy()
                minus = parameters.copy()
                plus[0, 0, 0] += np.pi / 2
                minus[0, 0, 0] -= np.pi / 2
                gradients.append(0.5 * (float(circuit(plus)) - float(circuit(minus))))
            records.append(
                {
                    "qubits": qubits,
                    "depth": depth,
                    "seeds": config["gradient_seeds"],
                    "gradient_mean": statistics.mean(gradients),
                    "gradient_variance": statistics.variance(gradients),
                    "gradients": gradients,
                    "circuit_executions": 2 * config["gradient_seeds"],
                    "observable": "global tensor product of Pauli-Z",
                }
            )
    return {"records": records, "interpretation": "small random-circuit trend only; not proof of a general barren-plateau law"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = ROOT / "results" / "track_c_quantum" / run_id
    run_dir.mkdir(parents=True)
    dump(run_dir / "config.json", config)

    qubo_path = ROOT / "data" / "track_b_qubo.json"
    payload = load_qubo(qubo_path)
    hamiltonian_check = verify_hamiltonian_basis(payload["qubo"])
    if not hamiltonian_check["passed"]:
        raise AssertionError("QUBO/Ising/Hamiltonian convention mismatch")
    qaoa = qaoa_experiment(payload, config)
    vqe = vqe_experiment(config)
    qnn = qnn_experiment(config)
    gradient_variance = gradient_variance_experiment(config)

    artifacts = {
        "qubo_source": {
            "path": "data/track_b_qubo.json",
            "sha256": hashlib.sha256(qubo_path.read_bytes()).hexdigest(),
            "instruction": payload["track_c_instruction"],
        },
        "hamiltonian_check": hamiltonian_check,
        "qaoa": qaoa,
        "vqe": vqe,
        "qnn": qnn,
        "gradient_variance": gradient_variance,
    }
    dump(run_dir / "raw_results.json", artifacts)

    shot_summary = []
    for shots in config["shot_budgets"]:
        group = [item for item in qaoa["finite_shots"] if item["shots"] == shots]
        shot_summary.append(
            {
                "shots": shots,
                "mean_energy": statistics.mean(item["expected_energy"] for item in group),
                "energy_stdev": statistics.stdev(item["expected_energy"] for item in group) if len(group) > 1 else 0.0,
                "mean_feasible_rate": statistics.mean(item["feasible_sample_rate"] for item in group),
                "optimum_seen_all_seeds": all(
                    item["best_feasible"] is not None
                    and item["best_feasible"]["bits"] == [1, 1, 0, 0, 0]
                    for item in group
                ),
            }
        )
    noisy_group = qaoa["noisy"]
    summary = {
        "run_kind": config["run_kind"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pennylane": qml.__version__,
            "execution": "local ideal and noisy simulators only; no real quantum hardware",
        },
        "qubo": {
            "states_checked": 32,
            "maximum_basis_energy_error": hamiltonian_check["maximum_error"],
            "classical_optimum_energy": payload["classical_optimum"]["minimum_energy"],
            "classical_optimum_bits": [1, 1, 0, 0, 0],
        },
        "qaoa": {
            "best_analytic_energy": qaoa["best_run"]["final_expected_energy"],
            "best_seed": qaoa["best_run"]["seed"],
            "shot_summary": shot_summary,
            "noise_probability": config["noise_probability"],
            "noisy_mean_energy": statistics.mean(item["expected_energy"] for item in noisy_group),
            "noisy_mean_feasible_rate": statistics.mean(item["feasible_sample_rate"] for item in noisy_group),
            "noisy_optimum_seen_all_seeds": all(
                item["best_feasible"] is not None and item["best_feasible"]["bits"] == [1, 1, 0, 0, 0]
                for item in noisy_group
            ),
        },
        "vqe": {
            "exact_ground_energy": vqe["exact_ground_energy"],
            "optimizers": [
                {key: value for key, value in item.items() if key not in {"history", "parameters", "metric_condition_numbers"}}
                for item in vqe["optimizers"]
            ],
            "parameter_shift_absolute_error": vqe["derivative_check"]["absolute_error"],
        },
        "qnn": {
            "mean_train_accuracy": statistics.mean(item["train_accuracy"] for item in qnn["runs"]),
            "mean_test_accuracy": statistics.mean(item["test_accuracy"] for item in qnn["runs"]),
            "parameter_count": qnn["runs"][0]["parameter_count"],
            "mean_circuit_executions": statistics.mean(item["circuit_executions"] for item in qnn["runs"]),
        },
        "gradient_variance": [
            {key: value for key, value in item.items() if key != "gradients"}
            for item in gradient_variance["records"]
        ],
    }
    dump(run_dir / "summary.json", summary)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for run in qaoa["optimization_runs"]:
        axes[0].plot([item["energy"] for item in run["history"]], label=f"seed {run['seed']}")
    axes[0].axhline(-18, color="black", linestyle="--", label="classical optimum")
    axes[0].set_title("QAOA expected energy")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Energy")
    axes[0].legend(fontsize=8)
    for item in vqe["optimizers"]:
        axes[1].plot(item["history"], label=item["method"].replace("_", " "))
    axes[1].axhline(vqe["exact_ground_energy"], color="black", linestyle="--", label="exact")
    axes[1].set_title("VQE optimizer convergence")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Energy")
    axes[1].legend(fontsize=8)
    for depth in config["gradient_depths"]:
        group = [item for item in gradient_variance["records"] if item["depth"] == depth]
        axes[2].plot(
            [item["qubits"] for item in group],
            [item["gradient_variance"] for item in group],
            marker="o",
            label=f"depth {depth}",
        )
    axes[2].set_yscale("log")
    axes[2].set_title("Random-circuit gradient variance")
    axes[2].set_xlabel("Qubits")
    axes[2].set_ylabel("Variance (log scale)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "quantum_optimization_comparison.png", dpi=160)
    plt.close(fig)

    dump(run_dir / "status.json", {"passed": True, "run_directory": run_dir.relative_to(ROOT).as_posix()})
    dump(
        ROOT / "results" / "track_c_quantum" / "latest-success.json",
        {
            "passed": True,
            "run_kind": config["run_kind"],
            "run_directory": run_dir.relative_to(ROOT).as_posix(),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
