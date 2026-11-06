import numpy as np

from optimization_lab.parallel_optimization import (
    Task,
    coverage_value,
    exhaustive_max_coverage,
    greedy_max_coverage,
    run_data_parallel,
    run_model_parallel,
    run_pipeline,
    run_tasks,
    verify_submodular_axioms,
)


def test_max_coverage_and_axioms():
    sets = [frozenset({0, 1}), frozenset({1, 2}), frozenset({3}), frozenset({2, 3})]
    checks = verify_submodular_axioms(sets)
    assert all(checks[name] for name in ("normalized", "nonnegative", "monotone", "diminishing_returns"))
    greedy, greedy_value = greedy_max_coverage(sets, 2)
    optimum, optimum_value = exhaustive_max_coverage(sets, 2)
    assert coverage_value(greedy, sets) == greedy_value == optimum_value
    assert coverage_value(optimum, sets) == optimum_value


def test_task_schedulers_preserve_outputs():
    tasks = [Task(index, 1000 + index * 10, index % 2, 17) for index in range(6)]
    baseline = run_tasks(tasks, "serial", 1)
    for method in ("static", "dynamic", "work_stealing"):
        result = run_tasks(tasks, method, 2)
        assert result["outputs"] == baseline["outputs"]


def test_parallelism_demonstrations_preserve_outputs():
    values = list(range(1000))
    assert run_data_parallel(values, 1)["result"] == run_data_parallel(values, 2)["result"]

    rng = np.random.default_rng(7)
    features = rng.normal(size=(40, 8))
    weights = rng.normal(size=8)
    serial = run_model_parallel(features, weights, 1)
    parallel = run_model_parallel(features, weights, 2)
    assert np.allclose(serial["prediction"], parallel["prediction"], rtol=1e-12, atol=1e-12)

    batches = [list(range(index * 10, index * 10 + 10)) for index in range(4)]
    assert run_pipeline(batches, False)["outputs"] == run_pipeline(batches, True)["outputs"]
