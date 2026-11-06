from pathlib import Path

import numpy as np
import pennylane as qml
import pytest

from optimization_lab.quantum_optimization import (
    analyze_counts,
    finite_difference_derivative,
    load_qubo,
    parameter_shift_derivative,
    qubo_energy,
    verify_hamiltonian_basis,
)


ROOT = Path(__file__).resolve().parents[1]


def test_qubo_hamiltonian_matches_all_basis_states():
    payload = load_qubo(ROOT / "data" / "track_b_qubo.json")
    verification = verify_hamiltonian_basis(payload["qubo"])
    assert verification["passed"]
    assert len(verification["states"]) == 32
    assert qubo_energy([1, 1, 0, 0, 0], payload["qubo"]) == -18


def test_parameter_shift_matches_finite_difference():
    device = qml.device("default.qubit", wires=1)

    @qml.qnode(device)
    def circuit(parameters):
        qml.RY(parameters[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    parameters = np.array([0.37])
    shifted = parameter_shift_derivative(circuit, parameters, 0)
    finite = finite_difference_derivative(circuit, parameters, 0)
    assert shifted == pytest.approx(finite, abs=1e-7)
    assert shifted == pytest.approx(-np.sin(0.37), abs=1e-7)


def test_count_analysis_preserves_bit_order_and_feasibility():
    payload = load_qubo(ROOT / "data" / "track_b_qubo.json")
    result = analyze_counts({"11000": 3, "00000": 1}, payload)
    assert result["best_feasible"]["bits"] == [1, 1, 0, 0, 0]
    assert result["feasible_sample_rate"] == 0.75
    assert result["expected_energy"] == pytest.approx((3 * -18 + 160) / 4)
