import numpy as np
import pytest
import torch

from optimization_lab.distributed_multitask import (
    fedavg_train,
    gradnorm_weight_step,
    mgda_two_task_weight,
    weighted_average,
)


def test_weighted_average_uses_sample_counts():
    result = weighted_average([np.array([0.0, 2.0]), np.array([10.0, 6.0])], [1, 3])
    assert np.allclose(result, [7.5, 5.0])


def test_fedavg_counts_bytes_and_updates():
    clients = [
        (np.array([[0.0], [1.0]]), np.array([0.0, 1.0])),
        (np.array([[2.0]]), np.array([1.0])),
    ]
    initial = np.zeros(2)
    _, metadata = fedavg_train(initial, clients, rounds=3, local_steps=2, learning_rate=0.1)
    assert metadata["client_parameter_updates"] == 12
    assert metadata["examples_processed"] == 18
    assert metadata["transmitted_parameter_bytes"] == 3 * 2 * initial.nbytes * 2


def test_mgda_known_two_gradient_cases():
    assert mgda_two_task_weight(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])) == pytest.approx(0.5)
    assert mgda_two_task_weight(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0])) == pytest.approx(0.8)


def test_gradnorm_weights_receive_gradient_and_normalize():
    updated, gradient, _ = gradnorm_weight_step(
        torch.ones(2),
        torch.tensor([2.0, 0.5]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 2.0]),
        learning_rate=0.05,
        alpha=1.5,
    )
    assert torch.any(gradient.abs() > 0)
    assert float(updated.sum()) == pytest.approx(2.0)
    assert torch.all(updated > 0)
