"""Quantum optimization primitives with explicit sign and cost conventions."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pennylane as qml


def load_qubo(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    qubo = payload["qubo"]
    if qubo["convention"] != "E(x)=offset+sum_i linear[i] x_i+sum_{i<j} quadratic[i,j] x_i x_j":
        raise ValueError("unexpected QUBO convention")
    if qubo["binary_to_spin"] != "x_i=(1-z_i)/2 with z_i in {-1,+1}":
        raise ValueError("unexpected binary-to-spin convention")
    return payload


def qubo_energy(bits: Sequence[int], qubo: dict) -> float:
    if len(bits) != qubo["n"] or any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must be a binary vector of QUBO dimension")
    energy = float(qubo["offset"])
    energy += sum(float(coefficient) * bit for coefficient, bit in zip(qubo["linear"], bits))
    for key, coefficient in qubo["quadratic"].items():
        left, right = map(int, key.split(","))
        energy += float(coefficient) * bits[left] * bits[right]
    return energy


def ising_energy(bits: Sequence[int], qubo: dict) -> float:
    spins = [1 - 2 * bit for bit in bits]
    ising = qubo["ising"]
    energy = float(ising["constant"])
    energy += sum(float(coefficient) * spin for coefficient, spin in zip(ising["h"], spins))
    for key, coefficient in ising["J"].items():
        left, right = map(int, key.split(","))
        energy += float(coefficient) * spins[left] * spins[right]
    return energy


def cost_hamiltonian(qubo: dict) -> qml.Hamiltonian:
    ising = qubo["ising"]
    coefficients = [float(ising["constant"])]
    operators = [qml.Identity(0)]
    for wire, coefficient in enumerate(ising["h"]):
        coefficients.append(float(coefficient))
        operators.append(qml.PauliZ(wire))
    for key, coefficient in ising["J"].items():
        left, right = map(int, key.split(","))
        coefficients.append(float(coefficient))
        operators.append(qml.PauliZ(left) @ qml.PauliZ(right))
    return qml.Hamiltonian(coefficients, operators)


def verify_hamiltonian_basis(qubo: dict) -> dict:
    hamiltonian = cost_hamiltonian(qubo)
    matrix = qml.matrix(hamiltonian, wire_order=range(qubo["n"]))
    diagonal = np.real(np.diag(matrix))
    maximum_error = 0.0
    records = []
    for index, bits in enumerate(product((0, 1), repeat=qubo["n"])):
        q_energy = qubo_energy(bits, qubo)
        i_energy = ising_energy(bits, qubo)
        h_energy = float(diagonal[index])
        maximum_error = max(maximum_error, abs(q_energy - i_energy), abs(q_energy - h_energy))
        records.append({"bits": list(bits), "qubo": q_energy, "ising": i_energy, "hamiltonian": h_energy})
    return {"passed": maximum_error < 1e-9, "maximum_error": maximum_error, "states": records}


def apply_qaoa(parameters, qubo: dict, noise_probability: float = 0.0) -> None:
    n = qubo["n"]
    for wire in range(n):
        qml.Hadamard(wire)
    for gamma, beta in parameters:
        for wire, coefficient in enumerate(qubo["ising"]["h"]):
            qml.RZ(2.0 * gamma * float(coefficient), wires=wire)
        for key, coefficient in qubo["ising"]["J"].items():
            left, right = map(int, key.split(","))
            qml.MultiRZ(2.0 * gamma * float(coefficient), wires=[left, right])
        for wire in range(n):
            qml.RX(2.0 * beta, wires=wire)
        if noise_probability:
            for wire in range(n):
                qml.DepolarizingChannel(noise_probability, wires=wire)


def analyze_counts(counts: dict, payload: dict) -> dict:
    qubo = payload["qubo"]
    total = int(sum(int(count) for count in counts.values()))
    expected_energy = 0.0
    feasible = 0
    best_feasible = None
    states = []
    for bitstring, count_value in sorted(counts.items()):
        bits = [int(value) for value in str(bitstring)]
        count = int(count_value)
        energy = qubo_energy(bits, qubo)
        is_feasible = sum(bits) == qubo["cardinality"]
        score = -energy if is_feasible else None
        expected_energy += count * energy / total
        feasible += count if is_feasible else 0
        if is_feasible and (best_feasible is None or energy < best_feasible["energy"]):
            best_feasible = {"bits": bits, "energy": energy, "score": score}
        states.append(
            {"bits": bits, "count": count, "probability": count / total, "energy": energy, "feasible": is_feasible}
        )
    return {
        "shots": total,
        "expected_energy": expected_energy,
        "feasible_sample_rate": feasible / total,
        "best_feasible": best_feasible,
        "states": states,
    }


def vqe_hamiltonian() -> qml.Hamiltonian:
    return qml.Hamiltonian(
        [-1.05, 0.5, 0.2],
        [qml.PauliZ(0), qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)],
    )


def apply_vqe_ansatz(parameters) -> None:
    qml.RY(parameters[0], wires=0)
    qml.RZ(parameters[1], wires=0)
    qml.RY(parameters[2], wires=1)
    qml.RZ(parameters[3], wires=1)
    qml.CNOT(wires=[0, 1])


def parameter_shift_derivative(function, parameters: np.ndarray, index: int) -> float:
    plus = np.array(parameters, dtype=float, copy=True)
    minus = np.array(parameters, dtype=float, copy=True)
    plus[index] += np.pi / 2
    minus[index] -= np.pi / 2
    return 0.5 * (float(function(plus)) - float(function(minus)))


def finite_difference_derivative(function, parameters: np.ndarray, index: int, epsilon: float = 1e-6) -> float:
    plus = np.array(parameters, dtype=float, copy=True)
    minus = np.array(parameters, dtype=float, copy=True)
    plus[index] += epsilon
    minus[index] -= epsilon
    return (float(function(plus)) - float(function(minus))) / (2 * epsilon)


def score_bitstring(bits: Sequence[int], source_problem: dict) -> float:
    score = sum(benefit * bit for benefit, bit in zip(source_problem["benefits"], bits))
    synergies = source_problem["synergies"]
    for left in range(len(bits)):
        for right in range(left + 1, len(bits)):
            score += synergies[left][right] * bits[left] * bits[right]
    return float(score)
