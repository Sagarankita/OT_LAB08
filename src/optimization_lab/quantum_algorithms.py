"""Small, explicit quantum algorithms and QML primitives for Track I."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

import numpy as np
import pennylane as qml


def apply_deutsch_jozsa(input_wires: Sequence[int], ancilla: int, balanced: bool) -> None:
    """Direct DJ circuit; balanced oracle is parity over all input bits."""
    qml.PauliX(ancilla)
    for wire in [*input_wires, ancilla]:
        qml.Hadamard(wire)
    if balanced:
        for wire in input_wires:
            qml.CNOT(wires=[wire, ancilla])
    for wire in input_wires:
        qml.Hadamard(wire)


def apply_bernstein_vazirani(secret: Sequence[int], input_wires: Sequence[int], ancilla: int) -> None:
    """Direct BV circuit with a CNOT oracle for the hidden bit string."""
    if len(secret) != len(input_wires) or any(bit not in (0, 1) for bit in secret):
        raise ValueError("secret must be a binary vector matching input wires")
    qml.PauliX(ancilla)
    for wire in [*input_wires, ancilla]:
        qml.Hadamard(wire)
    for bit, wire in zip(secret, input_wires):
        if bit:
            qml.CNOT(wires=[wire, ancilla])
    for wire in input_wires:
        qml.Hadamard(wire)


def _phase_flip_basis_state(marked: Sequence[int], wires: Sequence[int]) -> None:
    for bit, wire in zip(marked, wires):
        if bit == 0:
            qml.PauliX(wire)
    qml.Hadamard(wires[-1])
    qml.MultiControlledX(wires=[*wires[:-1], wires[-1]])
    qml.Hadamard(wires[-1])
    for bit, wire in zip(marked, wires):
        if bit == 0:
            qml.PauliX(wire)


def apply_grover(marked: Sequence[int], wires: Sequence[int], iterations: int) -> None:
    """Direct phase oracle and diffusion operator for one marked item."""
    if len(marked) != len(wires):
        raise ValueError("marked bit string must match wires")
    for wire in wires:
        qml.Hadamard(wire)
    for _ in range(iterations):
        _phase_flip_basis_state(marked, wires)
        for wire in wires:
            qml.Hadamard(wire)
            qml.PauliX(wire)
        qml.Hadamard(wires[-1])
        qml.MultiControlledX(wires=[*wires[:-1], wires[-1]])
        qml.Hadamard(wires[-1])
        for wire in wires:
            qml.PauliX(wire)
            qml.Hadamard(wire)


def grover_iteration_count(qubits: int) -> int:
    return max(1, int(math.floor(math.pi / 4 * math.sqrt(2**qubits))))


def modular_multiplication_unitary(base: int, modulus: int, qubits: int) -> np.ndarray:
    """Permutation unitary |x> -> |base*x mod modulus> for x<modulus."""
    dimension = 2**qubits
    if math.gcd(base, modulus) != 1 or modulus > dimension:
        raise ValueError("base must be coprime and modulus must fit the register")
    matrix = np.zeros((dimension, dimension), dtype=complex)
    for value in range(dimension):
        mapped = (base * value) % modulus if value < modulus else value
        matrix[mapped, value] = 1.0
    if not np.allclose(matrix.conj().T @ matrix, np.eye(dimension)):
        raise AssertionError("modular multiplication matrix is not unitary")
    return matrix


def infer_order_from_phase(phase_index: int, estimation_qubits: int, base: int, modulus: int) -> int | None:
    fraction = Fraction(phase_index, 2**estimation_qubits).limit_denominator(modulus)
    candidate = fraction.denominator
    if candidate > 0 and pow(base, candidate, modulus) == 1:
        return candidate
    for multiple in range(2, modulus + 1):
        order = candidate * multiple
        if order < 2 * modulus and pow(base, order, modulus) == 1:
            return order
    return None


def shor_factors_from_order(base: int, modulus: int, order: int) -> tuple[int, int] | None:
    if order % 2:
        return None
    value = pow(base, order // 2, modulus)
    left = math.gcd(value - 1, modulus)
    right = math.gcd(value + 1, modulus)
    if left in (1, modulus) or right in (1, modulus):
        return None
    return tuple(sorted((left, right)))


def annealing_hamiltonians(qubits: int) -> tuple[qml.Hamiltonian, qml.Hamiltonian]:
    driver = qml.Hamiltonian([-1.0] * qubits, [qml.PauliX(wire) for wire in range(qubits)])
    problem = qml.Hamiltonian(
        [1.0] * (qubits - 1),
        [qml.PauliZ(wire) @ qml.PauliZ(wire + 1) for wire in range(qubits - 1)],
    )
    return driver, problem


def apply_quantum_annealing(qubits: int, total_time: float, steps: int) -> None:
    # Hamiltonian observables are metadata here, not circuit operations.
    with qml.QueuingManager.stop_recording():
        driver, problem = annealing_hamiltonians(qubits)
    for wire in range(qubits):
        qml.Hadamard(wire)
    delta = total_time / steps
    for step in range(steps):
        schedule = (step + 0.5) / steps
        with qml.QueuingManager.stop_recording():
            hamiltonian = (1.0 - schedule) * driver + schedule * problem
        qml.ApproxTimeEvolution(hamiltonian, delta, 1)


def chain_ising_energy(bits: Sequence[int]) -> int:
    spins = [1 - 2 * bit for bit in bits]
    return sum(spins[index] * spins[index + 1] for index in range(len(spins) - 1))


def apply_feature_map(inputs: Sequence[float], wires: Sequence[int]) -> None:
    for value, wire in zip(inputs, wires):
        qml.Hadamard(wire)
        qml.RZ(float(value), wires=wire)
        qml.RY(float(value), wires=wire)
    for left, right in zip(wires[:-1], wires[1:]):
        qml.CZ(wires=[left, right])


def apply_variational_classifier(inputs, weights, noise_probability: float = 0.0) -> None:
    qml.AngleEmbedding(inputs, wires=[0, 1], rotation="Y")
    for layer in range(weights.shape[0]):
        qml.RY(weights[layer, 0], wires=0)
        qml.RY(weights[layer, 1], wires=1)
        qml.CNOT(wires=[0, 1])
        if noise_probability:
            qml.DepolarizingChannel(noise_probability, wires=0)
            qml.DepolarizingChannel(noise_probability, wires=1)
