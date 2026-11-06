"""Verified discrete-optimization models used by Track B."""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class Item:
    value: int
    weight: int


@dataclass(frozen=True)
class KnapsackSolution:
    indices: tuple[int, ...]
    total_value: int
    total_weight: int
    details: dict[str, Any]


def validate_knapsack(items: Sequence[Item], capacity: int) -> None:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
        raise ValueError("capacity must be a non-negative integer")
    for index, item in enumerate(items):
        if isinstance(item.value, bool) or not isinstance(item.value, int) or item.value < 0:
            raise ValueError(f"item {index} value must be a non-negative integer")
        if isinstance(item.weight, bool) or not isinstance(item.weight, int) or item.weight <= 0:
            raise ValueError(f"item {index} weight must be a positive integer")


def evaluate_selection(items: Sequence[Item], indices: Iterable[int]) -> tuple[int, int]:
    chosen = tuple(indices)
    if len(set(chosen)) != len(chosen):
        raise ValueError("selected indices must be unique")
    if any(isinstance(index, bool) or not isinstance(index, int)
           or index < 0 or index >= len(items) for index in chosen):
        raise ValueError("selected index out of range")
    return (sum(items[index].value for index in chosen),
            sum(items[index].weight for index in chosen))


def _solution(items: Sequence[Item], indices: Iterable[int], **details: Any) -> KnapsackSolution:
    normalized = tuple(sorted(indices))
    value, weight = evaluate_selection(items, normalized)
    return KnapsackSolution(normalized, value, weight, details)


def greedy_knapsack(items: Sequence[Item], capacity: int) -> KnapsackSolution:
    validate_knapsack(items, capacity)
    order = sorted(range(len(items)), key=lambda i: (
        -items[i].value / items[i].weight, -items[i].value, items[i].weight, i
    ))
    chosen: list[int] = []
    remaining = capacity
    for index in order:
        if items[index].weight <= remaining:
            chosen.append(index)
            remaining -= items[index].weight
    return _solution(items, chosen, ordering="value_weight_ratio", exact=False)


def dynamic_programming_knapsack(items: Sequence[Item], capacity: int) -> KnapsackSolution:
    validate_knapsack(items, capacity)
    # Each cell stores (value, chosen original indices); descending capacity
    # ensures every item is used at most once.
    table: list[tuple[int, tuple[int, ...]]] = [(0, ()) for _ in range(capacity + 1)]
    for index, item in enumerate(items):
        for weight in range(capacity, item.weight - 1, -1):
            previous_value, previous_indices = table[weight - item.weight]
            candidate = (previous_value + item.value, previous_indices + (index,))
            current = table[weight]
            if candidate[0] > current[0] or (candidate[0] == current[0] and candidate[1] < current[1]):
                table[weight] = candidate
    best_weight, best = min(
        enumerate(table), key=lambda pair: (-pair[1][0], pair[0], pair[1][1])
    )
    result = _solution(items, best[1], exact=True, table_cells=len(table) * (len(items) + 1))
    if result.total_weight > best_weight:
        raise AssertionError("DP reconstruction weight exceeds its table capacity")
    return result


def ilp_knapsack(items: Sequence[Item], capacity: int) -> KnapsackSolution:
    validate_knapsack(items, capacity)
    if not items:
        return _solution(items, (), exact=True, solver="scipy.optimize.milp", status=0,
                         message="empty instance", mip_gap=0.0)
    values = np.array([item.value for item in items], dtype=float)
    weights = np.array([item.weight for item in items], dtype=float)
    result = milp(
        c=-values,
        integrality=np.ones(len(items), dtype=int),
        bounds=Bounds(np.zeros(len(items)), np.ones(len(items))),
        constraints=LinearConstraint(weights.reshape(1, -1), -np.inf, float(capacity)),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"ILP failed with status {result.status}: {result.message}")
    rounded = np.rint(result.x).astype(int)
    if not np.allclose(result.x, rounded, atol=1e-6) or np.any((rounded < 0) | (rounded > 1)):
        raise RuntimeError("ILP returned a non-binary solution")
    chosen = tuple(int(i) for i in np.flatnonzero(rounded))
    return _solution(
        items, chosen, exact=True, solver="scipy.optimize.milp/HiGHS",
        status=int(result.status), message=str(result.message),
        mip_gap=float(getattr(result, "mip_gap", 0.0) or 0.0),
        mip_node_count=int(getattr(result, "mip_node_count", 0) or 0),
    )


def branch_and_bound_knapsack(items: Sequence[Item], capacity: int) -> KnapsackSolution:
    validate_knapsack(items, capacity)
    order = sorted(range(len(items)), key=lambda i: (
        -items[i].value / items[i].weight, -items[i].value, items[i].weight, i
    ))

    def bound(level: int, weight: int, value: int) -> float:
        if weight > capacity:
            return -math.inf
        upper = float(value)
        remaining = capacity - weight
        for position in range(level, len(order)):
            item = items[order[position]]
            if item.weight <= remaining:
                upper += item.value
                remaining -= item.weight
            else:
                upper += item.value * (remaining / item.weight)
                break
        return upper

    incumbent = greedy_knapsack(items, capacity)
    best_value, best_weight, best_indices = (
        incumbent.total_value, incumbent.total_weight, incumbent.indices
    )
    # Heap members: negative upper bound, level, current weight/value, positions chosen.
    queue: list[tuple[float, int, int, int, tuple[int, ...]]] = [
        (-bound(0, 0, 0), 0, 0, 0, ())
    ]
    nodes_expanded = 0
    nodes_pruned = 0
    while queue:
        negative_upper, level, weight, value, chosen = heapq.heappop(queue)
        upper = -negative_upper
        if upper + 1e-12 < best_value:
            nodes_pruned += 1
            continue
        nodes_expanded += 1
        if level == len(order):
            normalized = tuple(sorted(order[position] for position in chosen))
            if (value > best_value or
                (value == best_value and (weight < best_weight or
                 (weight == best_weight and normalized < best_indices)))):
                best_value, best_weight, best_indices = value, weight, normalized
            continue
        original_index = order[level]
        item = items[original_index]
        include_weight = weight + item.weight
        if include_weight <= capacity:
            include_value = value + item.value
            include_chosen = chosen + (level,)
            normalized = tuple(sorted(order[position] for position in include_chosen))
            if (include_value > best_value or
                (include_value == best_value and (include_weight < best_weight or
                 (include_weight == best_weight and normalized < best_indices)))):
                best_value, best_weight, best_indices = include_value, include_weight, normalized
            include_bound = bound(level + 1, include_weight, include_value)
            if include_bound + 1e-12 >= best_value:
                heapq.heappush(queue, (-include_bound, level + 1, include_weight,
                                      include_value, include_chosen))
            else:
                nodes_pruned += 1
        exclude_bound = bound(level + 1, weight, value)
        if exclude_bound + 1e-12 >= best_value:
            heapq.heappush(queue, (-exclude_bound, level + 1, weight, value, chosen))
        else:
            nodes_pruned += 1
    result = _solution(items, best_indices, exact=True, nodes_expanded=nodes_expanded,
                       nodes_pruned=nodes_pruned, initial_incumbent=incumbent.total_value)
    if result.total_value != best_value or result.total_weight != best_weight:
        raise AssertionError("Branch-and-Bound bookkeeping mismatch")
    return result


def exhaustive_knapsack(items: Sequence[Item], capacity: int, max_items: int = 24) -> KnapsackSolution:
    validate_knapsack(items, capacity)
    if len(items) > max_items:
        raise ValueError(f"exhaustive oracle limited to {max_items} items")
    best_value, best_weight, best_indices = 0, 0, ()
    feasible_subsets = 0
    for mask in range(1 << len(items)):
        indices = tuple(index for index in range(len(items)) if mask & (1 << index))
        value, weight = evaluate_selection(items, indices)
        if weight <= capacity:
            feasible_subsets += 1
            if (value > best_value or
                (value == best_value and (weight < best_weight or
                 (weight == best_weight and indices < best_indices)))):
                best_value, best_weight, best_indices = value, weight, indices
    return _solution(items, best_indices, exact=True, subsets=1 << len(items),
                     feasible_subsets=feasible_subsets)


def generate_knapsack(seed: int, item_count: int, capacity_ratio: float) -> tuple[list[Item], int]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if item_count <= 0:
        raise ValueError("item_count must be positive")
    if not 0 < capacity_ratio < 1:
        raise ValueError("capacity_ratio must be between zero and one")
    generator = np.random.default_rng(seed)
    weights = generator.integers(1, 31, size=item_count)
    values = generator.integers(1, 61, size=item_count)
    items = [Item(int(value), int(weight)) for value, weight in zip(values, weights)]
    capacity = max(1, int(math.floor(sum(item.weight for item in items) * capacity_ratio)))
    return items, capacity


def score_selection(bits: Sequence[int], benefits: Sequence[float],
                    synergies: Sequence[Sequence[float]]) -> float:
    _validate_binary_model(bits, benefits, synergies)
    return (sum(float(benefits[i]) * bits[i] for i in range(len(bits))) +
            sum(float(synergies[i][j]) * bits[i] * bits[j]
                for i in range(len(bits)) for j in range(i + 1, len(bits))))


def _validate_binary_model(bits: Sequence[int], benefits: Sequence[float],
                           synergies: Sequence[Sequence[float]]) -> None:
    n = len(benefits)
    if len(bits) != n or len(synergies) != n or any(len(row) != n for row in synergies):
        raise ValueError("binary model dimensions do not agree")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only zero or one")
    if any(abs(float(synergies[i][j]) - float(synergies[j][i])) > 1e-12
           for i in range(n) for j in range(n)):
        raise ValueError("synergy matrix must be symmetric")
    if any(abs(float(synergies[i][i])) > 1e-12 for i in range(n)):
        raise ValueError("synergy diagonal must be zero")


def sufficient_cardinality_penalty(benefits: Sequence[float],
                                   synergies: Sequence[Sequence[float]]) -> float:
    n = len(benefits)
    _validate_binary_model([0] * n, benefits, synergies)
    # With non-negative rewards, this upper-bounds every score. A penalty larger
    # than it makes every infeasible state positive, while any feasible state
    # has non-positive energy.
    if any(value < 0 for value in benefits) or any(synergies[i][j] < 0
           for i in range(n) for j in range(i + 1, n)):
        raise ValueError("this sufficient-bound proof requires non-negative rewards")
    upper = sum(float(value) for value in benefits) + sum(
        float(synergies[i][j]) for i in range(n) for j in range(i + 1, n)
    )
    return upper + 1.0


def build_cardinality_qubo(benefits: Sequence[float], synergies: Sequence[Sequence[float]],
                           cardinality: int, penalty: float) -> dict[str, Any]:
    n = len(benefits)
    _validate_binary_model([0] * n, benefits, synergies)
    if isinstance(cardinality, bool) or not isinstance(cardinality, int) or not 0 <= cardinality <= n:
        raise ValueError("cardinality must be an integer between zero and n")
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and non-negative")
    linear = [-float(benefits[i]) + penalty * (1 - 2 * cardinality) for i in range(n)]
    quadratic = {
        f"{i},{j}": -float(synergies[i][j]) + 2 * penalty
        for i in range(n) for j in range(i + 1, n)
    }
    offset = penalty * cardinality * cardinality
    pair_sum_by_variable = [0.0] * n
    for key, coefficient in quadratic.items():
        i, j = (int(value) for value in key.split(","))
        pair_sum_by_variable[i] += coefficient
        pair_sum_by_variable[j] += coefficient
    ising_h = [-0.5 * linear[i] - 0.25 * pair_sum_by_variable[i] for i in range(n)]
    ising_j = {key: 0.25 * value for key, value in quadratic.items()}
    ising_constant = offset + 0.5 * sum(linear) + 0.25 * sum(quadratic.values())
    return {
        "convention": "E(x)=offset+sum_i linear[i] x_i+sum_{i<j} quadratic[i,j] x_i x_j",
        "binary_to_spin": "x_i=(1-z_i)/2 with z_i in {-1,+1}",
        "n": n, "cardinality": cardinality, "penalty": float(penalty),
        "linear": linear, "quadratic": quadratic, "offset": offset,
        "ising": {"constant": ising_constant, "h": ising_h, "J": ising_j},
    }


def qubo_energy(bits: Sequence[int], qubo: dict[str, Any]) -> float:
    if len(bits) != qubo["n"] or any(bit not in (0, 1) for bit in bits):
        raise ValueError("invalid QUBO bit vector")
    return (float(qubo["offset"]) +
            sum(float(qubo["linear"][i]) * bits[i] for i in range(len(bits))) +
            sum(float(coefficient) * bits[int(key.split(",")[0])]
                * bits[int(key.split(",")[1])]
                for key, coefficient in qubo["quadratic"].items()))


def ising_energy(spins: Sequence[int], qubo: dict[str, Any]) -> float:
    if len(spins) != qubo["n"] or any(spin not in (-1, 1) for spin in spins):
        raise ValueError("invalid Ising spin vector")
    model = qubo["ising"]
    return (float(model["constant"]) +
            sum(float(model["h"][i]) * spins[i] for i in range(len(spins))) +
            sum(float(coefficient) * spins[int(key.split(",")[0])]
                * spins[int(key.split(",")[1])]
                for key, coefficient in model["J"].items()))


def enumerate_binary_model(benefits: Sequence[float], synergies: Sequence[Sequence[float]],
                           cardinality: int, penalty: float) -> dict[str, Any]:
    qubo = build_cardinality_qubo(benefits, synergies, cardinality, penalty)
    states = []
    for bits in itertools.product((0, 1), repeat=len(benefits)):
        score = score_selection(bits, benefits, synergies)
        expected = -score + penalty * (sum(bits) - cardinality) ** 2
        q_energy = qubo_energy(bits, qubo)
        spins = tuple(1 - 2 * bit for bit in bits)
        i_energy = ising_energy(spins, qubo)
        if not (math.isclose(expected, q_energy, abs_tol=1e-9) and
                math.isclose(q_energy, i_energy, abs_tol=1e-9)):
            raise AssertionError("QUBO/Ising conversion mismatch")
        states.append({
            "bits": list(bits), "spins": list(spins), "selected": sum(bits),
            "feasible": sum(bits) == cardinality, "score": score,
            "expected_energy": expected, "qubo_energy": q_energy,
            "ising_energy": i_energy,
        })
    minimum = min(state["qubo_energy"] for state in states)
    minimizers = [state for state in states if math.isclose(state["qubo_energy"], minimum, abs_tol=1e-9)]
    return {"qubo": qubo, "states": states, "minimum_energy": minimum,
            "minimizers": minimizers}

