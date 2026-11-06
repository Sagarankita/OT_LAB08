import itertools
import math

import pytest

from optimization_lab.discrete import (
    Item, branch_and_bound_knapsack, build_cardinality_qubo,
    dynamic_programming_knapsack, enumerate_binary_model, exhaustive_knapsack,
    generate_knapsack, greedy_knapsack, ilp_knapsack, ising_energy,
    qubo_energy, sufficient_cardinality_penalty,
)


def test_known_greedy_counterexample_and_exact_solvers():
    items = [Item(60, 10), Item(100, 20), Item(120, 30)]
    greedy = greedy_knapsack(items, 50)
    assert greedy.total_value == 160
    for solve in (dynamic_programming_knapsack, ilp_knapsack,
                  branch_and_bound_knapsack, exhaustive_knapsack):
        result = solve(items, 50)
        assert result.total_value == 220
        assert result.total_weight == 50
        assert result.indices == (1, 2)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_exact_solvers_match_exhaustive(seed):
    items, capacity = generate_knapsack(seed, 10, 0.35)
    oracle = exhaustive_knapsack(items, capacity)
    for solve in (dynamic_programming_knapsack, ilp_knapsack, branch_and_bound_knapsack):
        result = solve(items, capacity)
        assert result.total_value == oracle.total_value
        assert result.total_weight <= capacity


@pytest.mark.parametrize("items,capacity", [
    ([Item(1, 0)], 2), ([Item(-1, 1)], 2), ([Item(1, 1)], -1),
])
def test_invalid_knapsack_input(items, capacity):
    with pytest.raises(ValueError):
        greedy_knapsack(items, capacity)


def selection_model():
    benefits = [8, 7, 6, 5, 4]
    synergies = [
        [0, 3, 1, 0, 0], [3, 0, 2, 1, 0], [1, 2, 0, 1, 1],
        [0, 1, 1, 0, 0], [0, 0, 1, 0, 0],
    ]
    return benefits, synergies


def test_qubo_and_ising_match_every_state():
    benefits, synergies = selection_model()
    penalty = sufficient_cardinality_penalty(benefits, synergies)
    model = enumerate_binary_model(benefits, synergies, 2, penalty)
    assert len(model["states"]) == 32
    assert all(state["feasible"] for state in model["minimizers"])
    assert model["minimizers"][0]["bits"] == [1, 1, 0, 0, 0]
    for bits in itertools.product((0, 1), repeat=5):
        spins = [1 - 2 * bit for bit in bits]
        assert math.isclose(qubo_energy(bits, model["qubo"]),
                            ising_energy(spins, model["qubo"]), abs_tol=1e-9)


def test_insufficient_penalty_can_select_infeasible_state():
    benefits, synergies = selection_model()
    model = enumerate_binary_model(benefits, synergies, 2, 1)
    assert any(not state["feasible"] for state in model["minimizers"])


def test_sufficient_penalty_requires_nonnegative_reward_bound():
    with pytest.raises(ValueError, match="non-negative"):
        sufficient_cardinality_penalty([1, -1], [[0, 0], [0, 0]])
