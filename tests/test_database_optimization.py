import itertools
import sqlite3

import pytest

from optimization_lab.database_optimization import (
    CANDIDATES, QUERIES, connect, create_dataset, create_indexes,
    dynamic_programming_select, execute_query, greedy_select, index_storage,
    query_plan, rows_hash,
)


def test_small_dataset_queries_indexes_and_hashes(tmp_path):
    path = tmp_path / "small.db"
    counts = create_dataset(path, seed=9, order_count=2000, customer_count=200,
                            product_count=100)
    assert counts == {"customers": 200, "products": 100, "orders": 2000}
    connection = connect(path)
    baseline = {query.name: rows_hash(execute_query(connection, query)) for query in QUERIES}
    create_indexes(connection, [CANDIDATES[0].name, CANDIDATES[1].name])
    assert all(index_storage(connection, [CANDIDATES[0].name]).values())
    assert baseline == {query.name: rows_hash(execute_query(connection, query))
                        for query in QUERIES}
    assert all(query_plan(connection, query) for query in QUERIES)
    connection.close()


def test_selection_is_feasible_and_dp_matches_exhaustive():
    items = [
        {"name": "a", "pages": 4, "benefit_us": 7},
        {"name": "b", "pages": 3, "benefit_us": 6},
        {"name": "c", "pages": 2, "benefit_us": 4},
    ]
    budget = 6
    selected = dynamic_programming_select(items, budget)
    value = sum(item["benefit_us"] for item in items if item["name"] in selected)
    oracle = max(sum(item["benefit_us"] for item in subset)
                 for count in range(len(items) + 1)
                 for subset in itertools.combinations(items, count)
                 if sum(item["pages"] for item in subset) <= budget)
    assert value == oracle
    assert sum(item["pages"] for item in items if item["name"] in selected) <= budget
    greedy = greedy_select(items, budget)
    assert sum(item["pages"] for item in items if item["name"] in greedy) <= budget


def test_invalid_index_and_budget_are_rejected(tmp_path):
    path = tmp_path / "invalid.db"
    create_dataset(path, seed=1, order_count=10, customer_count=5, product_count=5)
    connection = connect(path)
    with pytest.raises(ValueError):
        create_indexes(connection, ["not_an_index"])
    with pytest.raises(ValueError):
        greedy_select([], -1)
    with pytest.raises(ValueError):
        dynamic_programming_select([], -1)
    connection.close()


def test_rows_hash_is_order_aware():
    assert rows_hash([(1,), (2,)]) != rows_hash([(2,), (1,)])
