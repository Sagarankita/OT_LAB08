#!/usr/bin/env python3
"""Run Track I quantum algorithms and matched QML comparisons."""

from __future__ import annotations

import argparse
import json
import math
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
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
from pennylane import numpy as pnp
from torch import nn

sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.quantum_algorithms import (  # noqa: E402
    annealing_hamiltonians,
    apply_bernstein_vazirani,
    apply_deutsch_jozsa,
    apply_feature_map,
    apply_grover,
    apply_quantum_annealing,
    apply_variational_classifier,
    chain_ising_energy,
    grover_iteration_count,
    infer_order_from_phase,
    modular_multiplication_unitary,
    shor_factors_from_order,
)


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resource_dict(specification: dict) -> dict:
    resources = specification["resources"]
    return {
        "num_wires": getattr(resources, "num_wires", getattr(specification, "num_device_wires", None)),
        "num_gates": resources.num_gates,
        "depth": resources.depth,
        "gate_types": dict(resources.gate_types),
    }


def algorithm_experiments(config: dict) -> dict:
    results = {}

    # Deutsch-Jozsa, direct implementation.
    dj_device = qml.device("default.qubit", wires=4, shots=config["algorithm_shots"], seed=3)

    @qml.qnode(dj_device)
    def dj_circuit(balanced: bool):
        apply_deutsch_jozsa([0, 1, 2], 3, balanced)
        return qml.counts(wires=[0, 1, 2])

    dj_constant = dj_circuit(False)
    dj_balanced = dj_circuit(True)
    results["deutsch_jozsa"] = {
        "implementation": "direct circuit and parity oracle",
        "constant_counts": {str(key): int(value) for key, value in dj_constant.items()},
        "balanced_counts": {str(key): int(value) for key, value in dj_balanced.items()},
        "constant_classified": set(map(str, dj_constant)) == {"000"},
        "balanced_classified": "000" not in set(map(str, dj_balanced)),
        "oracle_queries": 1,
        "resources": resource_dict(qml.specs(dj_circuit)(True)),
    }

    # Bernstein-Vazirani, direct implementation.
    secret = [1, 0, 1, 1]
    bv_device = qml.device("default.qubit", wires=5, shots=config["algorithm_shots"], seed=3)

    @qml.qnode(bv_device)
    def bv_circuit():
        apply_bernstein_vazirani(secret, [0, 1, 2, 3], 4)
        return qml.counts(wires=[0, 1, 2, 3])

    bv_counts = bv_circuit()
    results["bernstein_vazirani"] = {
        "implementation": "direct CNOT oracle circuit",
        "secret": secret,
        "counts": {str(key): int(value) for key, value in bv_counts.items()},
        "recovered_secret": max(bv_counts, key=bv_counts.get),
        "oracle_queries": 1,
        "classical_deterministic_queries": len(secret),
        "resources": resource_dict(qml.specs(bv_circuit)()),
    }

    # Grover with direct oracle and diffusion.
    marked = [1, 0, 1]
    iterations = grover_iteration_count(3)
    grover_device = qml.device("default.qubit", wires=3, shots=config["algorithm_shots"], seed=3)

    @qml.qnode(grover_device)
    def grover_counts():
        apply_grover(marked, [0, 1, 2], iterations)
        return qml.counts(wires=[0, 1, 2])

    analytic_device = qml.device("default.qubit", wires=3)

    @qml.qnode(analytic_device)
    def grover_probabilities():
        apply_grover(marked, [0, 1, 2], iterations)
        return qml.probs(wires=[0, 1, 2])

    quantum_times = []
    for _ in range(config["grover_timing_repetitions"]):
        started = time.perf_counter()
        grover_probabilities()
        quantum_times.append(time.perf_counter() - started)
    target_index = int("101", 2)

    def classical_linear_search() -> tuple[int, int]:
        queries = 0
        for candidate in range(8):
            queries += 1
            if candidate == target_index:
                return candidate, queries
        raise AssertionError

    classical_times = []
    actual_queries = None
    for _ in range(config["grover_timing_repetitions"]):
        started = time.perf_counter()
        _, actual_queries = classical_linear_search()
        classical_times.append(time.perf_counter() - started)
    counts = grover_counts()
    probabilities = grover_probabilities()
    results["grover"] = {
        "implementation": "direct phase oracle and diffusion circuit",
        "marked": marked,
        "iterations_and_oracle_queries": iterations,
        "success_probability": float(probabilities[target_index]),
        "counts": {str(key): int(value) for key, value in counts.items()},
        "classical_queries_for_fixed_target": actual_queries,
        "classical_expected_queries_uniform_target": 4.5,
        "classical_worst_case_queries": 8,
        "quantum_simulator_wall_seconds": {
            "mean": statistics.mean(quantum_times),
            "stdev": statistics.stdev(quantum_times),
        },
        "classical_wall_seconds": {
            "mean": statistics.mean(classical_times),
            "stdev": statistics.stdev(classical_times),
        },
        "resources": resource_dict(qml.specs(grover_probabilities)()),
    }

    # QFT and inverse-QFT recovery.
    qft_device = qml.device("default.qubit", wires=3)

    @qml.qnode(qft_device)
    def qft_state():
        qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
        qml.QFT(wires=[0, 1, 2])
        return qml.state()

    @qml.qnode(qft_device)
    def qft_roundtrip():
        qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
        qml.QFT(wires=[0, 1, 2])
        qml.adjoint(qml.QFT)(wires=[0, 1, 2])
        return qml.probs(wires=[0, 1, 2])

    qft_values = qft_state()
    expected_qft = np.array([np.exp(2j * np.pi * 5 * index / 8) / np.sqrt(8) for index in range(8)])
    # Global-phase invariant fidelity handles implementation phase convention.
    qft_fidelity = float(abs(np.vdot(expected_qft, qft_values)) ** 2)
    results["qft"] = {
        "input": "101",
        "state_real": np.real(qft_values).tolist(),
        "state_imag": np.imag(qft_values).tolist(),
        "analytic_state_fidelity": qft_fidelity,
        "roundtrip_probability": float(qft_roundtrip()[5]),
        "resources": resource_dict(qml.specs(qft_state)()),
    }

    # QPE on a known eigenphase 3/8.
    phase = 3 / 8
    qpe_unitary = np.diag([1.0, np.exp(2j * np.pi * phase)])
    qpe_device = qml.device("default.qubit", wires=4)

    @qml.qnode(qpe_device)
    def qpe_circuit():
        qml.PauliX(3)
        qml.QuantumPhaseEstimation(qpe_unitary, target_wires=[3], estimation_wires=[0, 1, 2])
        return qml.probs(wires=[0, 1, 2])

    qpe_probabilities = qpe_circuit()
    results["qpe"] = {
        "true_phase": phase,
        "estimated_index": int(np.argmax(qpe_probabilities)),
        "estimated_phase": int(np.argmax(qpe_probabilities)) / 8,
        "success_probability": float(np.max(qpe_probabilities)),
        "probabilities": qpe_probabilities.tolist(),
        "resources": resource_dict(qml.specs(qpe_circuit)()),
    }

    # Compiled order finding for N=15, a=2. The modular unitary is explicit, not a general Shor circuit.
    shor_unitary = modular_multiplication_unitary(2, 15, 4)
    shor_device = qml.device("default.qubit", wires=8)

    @qml.qnode(shor_device)
    def shor_order_finding():
        qml.BasisState(np.array([0, 0, 0, 1]), wires=[4, 5, 6, 7])
        qml.QuantumPhaseEstimation(
            shor_unitary, target_wires=[4, 5, 6, 7], estimation_wires=[0, 1, 2, 3]
        )
        return qml.probs(wires=[0, 1, 2, 3])

    shor_probabilities = shor_order_finding()
    peaks = [index for index, probability in enumerate(shor_probabilities) if probability > 0.1]
    orders = sorted(
        {
            order
            for index in peaks
            if index != 0
            for order in [infer_order_from_phase(index, 4, 2, 15)]
            if order is not None
        }
    )
    factors = shor_factors_from_order(2, 15, orders[0]) if orders else None
    results["shor_compiled_15"] = {
        "disclosure": "compiled fixed N=15, a=2 order-finding demonstration; not general factoring",
        "phase_probabilities": shor_probabilities.tolist(),
        "peak_indices": peaks,
        "inferred_orders": orders,
        "factors": factors,
        "successful_nonzero_phase_probability": float(sum(shor_probabilities[index] for index in peaks if index != 0)),
        "resources": resource_dict(qml.specs(shor_order_finding)()),
    }

    # Time-dependent unitary quantum annealing for a three-spin antiferromagnetic chain.
    annealing = []
    energies = [chain_ising_energy([int(bit) for bit in format(index, "03b")]) for index in range(8)]
    ground_energy = min(energies)
    ground_indices = [index for index, energy in enumerate(energies) if energy == ground_energy]
    for total_time in config["annealing_times"]:
        device = qml.device("default.qubit", wires=3)

        @qml.qnode(device)
        def annealing_circuit():
            apply_quantum_annealing(3, total_time, config["annealing_steps"])
            return qml.probs(wires=[0, 1, 2])

        probabilities = annealing_circuit()
        annealing.append(
            {
                "total_time": total_time,
                "trotter_steps": config["annealing_steps"],
                "probabilities": probabilities.tolist(),
                "probability_sum": float(sum(probabilities)),
                "ground_success_probability": float(sum(probabilities[index] for index in ground_indices)),
                "expected_problem_energy": float(sum(probability * energy for probability, energy in zip(probabilities, energies))),
                "circuit_model": "closed-system time-dependent unitary evolution with first-order Trotterization",
                "resources": resource_dict(qml.specs(annealing_circuit)()),
            }
        )
    results["quantum_annealing"] = {
        "problem": "three-spin antiferromagnetic Ising chain",
        "ground_energy": ground_energy,
        "ground_bitstrings": [format(index, "03b") for index in ground_indices],
        "runs": annealing,
        "hardware": "local gate-model simulation, not annealing hardware",
        "embedding_note": "real sparse-connectivity annealers may require chains of physical qubits; chain strength and broken-chain decoding add overhead",
    }
    return results


def make_qml_dataset() -> dict:
    rng = np.random.default_rng(2026)
    count = 20
    angles = np.linspace(0, np.pi, count)
    class_one = np.column_stack([np.cos(angles), np.sin(angles)])
    class_two = np.column_stack([1 - np.cos(angles), -np.sin(angles) - 0.5])
    features = np.vstack([class_one, class_two]) + 0.08 * rng.normal(size=(40, 2))
    labels = np.concatenate([np.ones(count), -np.ones(count)])
    order = rng.permutation(40)
    train, validation, test = order[:24], order[24:32], order[32:]
    mean = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    transformed = np.clip((features - mean) / scale, -2, 2) * np.pi / 2
    return {
        "features": transformed,
        "labels": labels,
        "train": train,
        "validation": validation,
        "test": test,
        "train_mean": mean,
        "train_scale": scale,
    }


def qml_experiments(config: dict) -> dict:
    data = make_qml_dataset()
    features, labels = data["features"], data["labels"]
    train, validation, test = data["train"], data["validation"], data["test"]

    variational_runs = []
    trained_weights = {}
    for seed in config["seeds"]:
        device = qml.device("default.qubit", wires=2)

        @qml.qnode(device, interface="autograd", diff_method="parameter-shift")
        def classifier(weights, inputs):
            apply_variational_classifier(inputs, weights)
            return qml.expval(qml.PauliZ(0))

        def cost(weights):
            predictions = pnp.stack([classifier(weights, features[index]) for index in train])
            return pnp.mean((predictions - labels[train]) ** 2)

        rng = np.random.default_rng(seed)
        weights = pnp.array(rng.normal(0, 0.2, size=(3, 2)), requires_grad=True)
        optimizer = qml.AdamOptimizer(config["qml_learning_rate"])
        history = []
        started = time.perf_counter()
        with qml.Tracker(device) as tracker:
            for epoch in range(config["qml_epochs"]):
                weights, old_cost = optimizer.step_and_cost(cost, weights)
                history.append(float(old_cost))
            predictions = {
                split: np.array([classifier(weights, features[index]) for index in indices])
                for split, indices in (("train", train), ("validation", validation), ("test", test))
            }
        accuracies = {
            split: float(np.mean(np.where(values >= 0, 1, -1) == labels[indices]))
            for (split, indices), values in zip(
                (("train", train), ("validation", validation), ("test", test)), predictions.values()
            )
        }
        trained_weights[seed] = np.asarray(weights)
        variational_runs.append(
            {
                "seed": seed,
                "parameter_count": int(np.size(weights)),
                "epochs": config["qml_epochs"],
                "accuracies": accuracies,
                "final_cost": float(cost(weights)),
                "runtime_seconds": time.perf_counter() - started,
                "circuit_executions": int(tracker.totals.get("executions", 0)),
                "history": history,
                "weights": np.asarray(weights).tolist(),
            }
        )

    # Noisy finite-shot evaluation reuses Track C's p=0.02 layerwise depolarizing model.
    best_seed = max(variational_runs, key=lambda item: item["accuracies"]["validation"])["seed"]
    noisy_results = []
    for noise_seed in config["seeds"]:
        device = qml.device("default.mixed", wires=2, shots=config["noise_shots"], seed=noise_seed)

        @qml.qnode(device)
        def noisy_classifier(inputs):
            apply_variational_classifier(
                inputs, trained_weights[best_seed], noise_probability=config["noise_probability"]
            )
            return qml.expval(qml.PauliZ(0))

        predictions = np.array([noisy_classifier(features[index]) for index in test])
        noisy_results.append(
            {
                "seed": noise_seed,
                "test_accuracy": float(np.mean(np.where(predictions >= 0, 1, -1) == labels[test])),
                "shots_per_prediction": config["noise_shots"],
                "circuit_evaluations": len(test),
                "noise": f"simulated layerwise depolarizing p={config['noise_probability']}",
            }
        )

    # Classical MLP with a comparable small parameter budget (13 versus 6).
    classical_runs = []
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor((labels + 1) / 2, dtype=torch.float32)
    torch.set_num_threads(1)
    for seed in config["seeds"]:
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        criterion = nn.BCEWithLogitsLoss()
        started = time.perf_counter()
        for _ in range(config["qml_epochs"]):
            optimizer.zero_grad()
            loss = criterion(model(x_tensor[train]).squeeze(-1), y_tensor[train])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            accuracies = {}
            for split, indices in (("train", train), ("validation", validation), ("test", test)):
                predictions = (model(x_tensor[indices]).squeeze(-1) >= 0).float()
                accuracies[split] = float((predictions == y_tensor[indices]).float().mean())
        classical_runs.append(
            {
                "seed": seed,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "epochs": config["qml_epochs"],
                "accuracies": accuracies,
                "runtime_seconds": time.perf_counter() - started,
                "circuit_executions": 0,
            }
        )

    # Quantum fidelity kernel with validation-only regularization selection.
    kernel_device = qml.device("default.qubit", wires=2)

    @qml.qnode(kernel_device)
    def kernel_circuit(left, right):
        apply_feature_map(left, [0, 1])
        qml.adjoint(apply_feature_map)(right, [0, 1])
        return qml.probs(wires=[0, 1])

    kernel_calls = 0

    def kernel_value(left, right):
        nonlocal kernel_calls
        kernel_calls += 1
        return float(kernel_circuit(left, right)[0])

    def symmetric_kernel(indices):
        matrix = np.empty((len(indices), len(indices)))
        for row in range(len(indices)):
            for column in range(row, len(indices)):
                value = kernel_value(features[indices[row]], features[indices[column]])
                matrix[row, column] = matrix[column, row] = value
        return matrix

    def cross_kernel(left_indices, right_indices):
        return np.array(
            [
                [kernel_value(features[left], features[right]) for right in right_indices]
                for left in left_indices
            ]
        )

    kernel_started = time.perf_counter()
    train_kernel = symmetric_kernel(train)
    validation_kernel = cross_kernel(validation, train)
    test_kernel = cross_kernel(test, train)
    best_kernel = None
    for regularization in config["kernel_lambdas"]:
        coefficients = np.linalg.solve(
            train_kernel + regularization * np.eye(len(train)), labels[train]
        )
        validation_accuracy = float(
            np.mean(np.where(validation_kernel @ coefficients >= 0, 1, -1) == labels[validation])
        )
        candidate = (validation_accuracy, -regularization, regularization, coefficients)
        if best_kernel is None or candidate[:2] > best_kernel[:2]:
            best_kernel = candidate
    _, _, selected_regularization, coefficients = best_kernel
    quantum_kernel_result = {
        "selected_regularization": selected_regularization,
        "validation_accuracy": best_kernel[0],
        "test_accuracy": float(np.mean(np.where(test_kernel @ coefficients >= 0, 1, -1) == labels[test])),
        "train_accuracy": float(np.mean(np.where(train_kernel @ coefficients >= 0, 1, -1) == labels[train])),
        "circuit_executions": kernel_calls,
        "runtime_seconds": time.perf_counter() - kernel_started,
        "train_kernel_min_eigenvalue": float(np.linalg.eigvalsh(train_kernel).min()),
    }

    # Classical RBF-kernel ridge baseline, tuned only on validation.
    def rbf(left, right, gamma):
        differences = left[:, None, :] - right[None, :, :]
        return np.exp(-gamma * np.sum(differences**2, axis=2))

    rbf_started = time.perf_counter()
    best_rbf = None
    for gamma in (0.1, 0.5, 1.0, 2.0):
        train_rbf = rbf(features[train], features[train], gamma)
        validation_rbf = rbf(features[validation], features[train], gamma)
        for regularization in config["kernel_lambdas"]:
            coefficients_rbf = np.linalg.solve(
                train_rbf + regularization * np.eye(len(train)), labels[train]
            )
            validation_accuracy = float(
                np.mean(np.where(validation_rbf @ coefficients_rbf >= 0, 1, -1) == labels[validation])
            )
            candidate = (
                validation_accuracy,
                -regularization,
                -gamma,
                gamma,
                regularization,
                coefficients_rbf,
                train_rbf,
            )
            if best_rbf is None or candidate[:3] > best_rbf[:3]:
                best_rbf = candidate
    test_rbf = rbf(features[test], features[train], best_rbf[3])
    classical_kernel_result = {
        "selected_gamma": best_rbf[3],
        "selected_regularization": best_rbf[4],
        "validation_accuracy": best_rbf[0],
        "test_accuracy": float(np.mean(np.where(test_rbf @ best_rbf[5] >= 0, 1, -1) == labels[test])),
        "train_accuracy": float(np.mean(np.where(best_rbf[6] @ best_rbf[5] >= 0, 1, -1) == labels[train])),
        "circuit_executions": 0,
        "runtime_seconds": time.perf_counter() - rbf_started,
    }

    return {
        "dataset": {
            "samples": 40,
            "train": train.tolist(),
            "validation": validation.tolist(),
            "test": test.tolist(),
            "preprocessing": "mean/std fitted on training split only, clipped then angle-scaled",
        },
        "variational_quantum_classifier": variational_runs,
        "noisy_quantum_classifier": noisy_results,
        "classical_mlp": classical_runs,
        "quantum_kernel": quantum_kernel_result,
        "classical_rbf_kernel": classical_kernel_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = ROOT / "results" / "track_i_quantum" / run_id
    run_dir.mkdir(parents=True)
    dump(run_dir / "config.json", config)
    algorithms = algorithm_experiments(config)
    qml_results = qml_experiments(config)
    raw = {"algorithms": algorithms, "qml": qml_results}
    dump(run_dir / "raw_results.json", raw)

    variational = qml_results["variational_quantum_classifier"]
    classical = qml_results["classical_mlp"]
    summary = {
        "run_kind": config["run_kind"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pennylane": qml.__version__,
            "torch": torch.__version__,
            "execution": "local ideal/noisy simulators only; no real quantum hardware",
        },
        "algorithm_success": {
            "deutsch_jozsa": algorithms["deutsch_jozsa"]["constant_classified"] and algorithms["deutsch_jozsa"]["balanced_classified"],
            "bernstein_vazirani": algorithms["bernstein_vazirani"]["recovered_secret"] == "1011",
            "grover_probability": algorithms["grover"]["success_probability"],
            "qft_fidelity": algorithms["qft"]["analytic_state_fidelity"],
            "qpe_success_probability": algorithms["qpe"]["success_probability"],
            "shor_factors": algorithms["shor_compiled_15"]["factors"],
            "annealing_slow_success": algorithms["quantum_annealing"]["runs"][-1]["ground_success_probability"],
        },
        "grover_comparison": {
            key: algorithms["grover"][key]
            for key in (
                "iterations_and_oracle_queries",
                "classical_queries_for_fixed_target",
                "classical_expected_queries_uniform_target",
                "classical_worst_case_queries",
                "success_probability",
                "quantum_simulator_wall_seconds",
                "classical_wall_seconds",
            )
        },
        "qml_comparison": {
            "variational_quantum_classifier": {
                "mean_train_accuracy": statistics.mean(item["accuracies"]["train"] for item in variational),
                "mean_validation_accuracy": statistics.mean(item["accuracies"]["validation"] for item in variational),
                "mean_test_accuracy": statistics.mean(item["accuracies"]["test"] for item in variational),
                "mean_runtime_seconds": statistics.mean(item["runtime_seconds"] for item in variational),
                "parameter_count": variational[0]["parameter_count"],
                "mean_circuit_executions": statistics.mean(item["circuit_executions"] for item in variational),
            },
            "classical_mlp": {
                "mean_train_accuracy": statistics.mean(item["accuracies"]["train"] for item in classical),
                "mean_validation_accuracy": statistics.mean(item["accuracies"]["validation"] for item in classical),
                "mean_test_accuracy": statistics.mean(item["accuracies"]["test"] for item in classical),
                "mean_runtime_seconds": statistics.mean(item["runtime_seconds"] for item in classical),
                "parameter_count": classical[0]["parameter_count"],
                "circuit_executions": 0,
            },
            "noisy_qnn_mean_test_accuracy": statistics.mean(
                item["test_accuracy"] for item in qml_results["noisy_quantum_classifier"]
            ),
            "quantum_kernel": qml_results["quantum_kernel"],
            "classical_rbf_kernel": qml_results["classical_rbf_kernel"],
        },
    }
    dump(run_dir / "summary.json", summary)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    names = ["DJ", "BV", "Grover", "QFT", "QPE", "Shor", "Anneal"]
    values = [
        1.0 if summary["algorithm_success"]["deutsch_jozsa"] else 0.0,
        1.0 if summary["algorithm_success"]["bernstein_vazirani"] else 0.0,
        summary["algorithm_success"]["grover_probability"],
        summary["algorithm_success"]["qft_fidelity"],
        summary["algorithm_success"]["qpe_success_probability"],
        1.0 if list(summary["algorithm_success"]["shor_factors"]) == [3, 5] else 0.0,
        summary["algorithm_success"]["annealing_slow_success"],
    ]
    axes[0].bar(names, values)
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_title("Algorithm known-answer success")
    axes[0].set_ylabel("Probability / pass indicator")
    axes[1].bar(
        ["Quantum\nsimulator", "Classical\nlinear"],
        [
            summary["grover_comparison"]["quantum_simulator_wall_seconds"]["mean"],
            summary["grover_comparison"]["classical_wall_seconds"]["mean"],
        ],
        color=["#8b5cf6", "#6b7280"],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Wall seconds (log scale)")
    axes[1].set_title("Grover query advantage ≠ simulator speed")
    qml_names = ["VQC", "Classical\nMLP", "Quantum\nkernel", "RBF\nkernel"]
    qml_values = [
        summary["qml_comparison"]["variational_quantum_classifier"]["mean_test_accuracy"],
        summary["qml_comparison"]["classical_mlp"]["mean_test_accuracy"],
        summary["qml_comparison"]["quantum_kernel"]["test_accuracy"],
        summary["qml_comparison"]["classical_rbf_kernel"]["test_accuracy"],
    ]
    axes[2].bar(qml_names, qml_values, color=["#8b5cf6", "#6b7280", "#a78bfa", "#9ca3af"])
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Test accuracy")
    axes[2].set_title("Same-data QML comparisons")
    fig.tight_layout()
    fig.savefig(run_dir / "quantum_algorithms_qml.png", dpi=160)
    plt.close(fig)

    dump(run_dir / "status.json", {"passed": True, "run_directory": run_dir.relative_to(ROOT).as_posix()})
    dump(
        ROOT / "results" / "track_i_quantum" / "latest-success.json",
        {"passed": True, "run_kind": config["run_kind"], "run_directory": run_dir.relative_to(ROOT).as_posix()},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
