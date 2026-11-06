import numpy as np
import pennylane as qml

from optimization_lab.quantum_algorithms import (
    apply_bernstein_vazirani,
    apply_deutsch_jozsa,
    apply_grover,
    chain_ising_energy,
    grover_iteration_count,
    infer_order_from_phase,
    modular_multiplication_unitary,
    shor_factors_from_order,
)


def test_direct_deutsch_jozsa_and_bv_circuits():
    device = qml.device("default.qubit", wires=5)

    @qml.qnode(device)
    def dj(balanced):
        apply_deutsch_jozsa([0, 1, 2], 3, balanced)
        return qml.probs(wires=[0, 1, 2])

    assert np.argmax(dj(False)) == 0
    assert np.argmax(dj(True)) != 0

    @qml.qnode(device)
    def bv():
        apply_bernstein_vazirani([1, 0, 1, 1], [0, 1, 2, 3], 4)
        return qml.probs(wires=[0, 1, 2, 3])

    assert np.argmax(bv()) == int("1011", 2)


def test_direct_grover_finds_marked_state():
    marked = [1, 0, 1]
    device = qml.device("default.qubit", wires=3)

    @qml.qnode(device)
    def circuit():
        apply_grover(marked, [0, 1, 2], grover_iteration_count(3))
        return qml.probs(wires=[0, 1, 2])

    probabilities = circuit()
    assert np.argmax(probabilities) == int("101", 2)
    assert probabilities[int("101", 2)] > 0.9


def test_compiled_shor_order_and_factors():
    unitary = modular_multiplication_unitary(2, 15, 4)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(16))
    assert infer_order_from_phase(4, 4, 2, 15) == 4
    assert shor_factors_from_order(2, 15, 4) == (3, 5)


def test_chain_ising_ground_states():
    energies = {format(index, "03b"): chain_ising_energy([int(bit) for bit in format(index, "03b")]) for index in range(8)}
    minimum = min(energies.values())
    assert minimum == -2
    assert {bits for bits, energy in energies.items() if energy == minimum} == {"010", "101"}
