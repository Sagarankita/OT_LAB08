"""FedAvg and multi-task optimization methods for Track D."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def logistic_loss_gradient(parameters: np.ndarray, features: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    augmented = np.column_stack([features, np.ones(len(features))])
    probabilities = sigmoid(augmented @ parameters)
    eps = 1e-12
    loss = -np.mean(labels * np.log(probabilities + eps) + (1 - labels) * np.log(1 - probabilities + eps))
    gradient = augmented.T @ (probabilities - labels) / len(labels)
    return float(loss), gradient


def evaluate_logistic(parameters: np.ndarray, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    loss, _ = logistic_loss_gradient(parameters, features, labels)
    augmented = np.column_stack([features, np.ones(len(features))])
    predictions = sigmoid(augmented @ parameters) >= 0.5
    return {"loss": loss, "accuracy": float(np.mean(predictions == labels))}


def weighted_average(parameters: Sequence[np.ndarray], sample_counts: Sequence[int]) -> np.ndarray:
    if len(parameters) != len(sample_counts) or not parameters:
        raise ValueError("parameters and sample counts must have equal nonzero length")
    if any(count <= 0 for count in sample_counts):
        raise ValueError("sample counts must be positive")
    total = sum(sample_counts)
    return sum(parameter * (count / total) for parameter, count in zip(parameters, sample_counts))


def local_logistic_train(
    parameters: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    steps: int,
    learning_rate: float,
) -> np.ndarray:
    result = parameters.copy()
    for _ in range(steps):
        _, gradient = logistic_loss_gradient(result, features, labels)
        result -= learning_rate * gradient
    return result


def centralized_train(
    initial: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    steps: int,
    learning_rate: float,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    result = local_logistic_train(initial, features, labels, steps, learning_rate)
    return result, time.perf_counter() - started


def fedavg_train(
    initial: np.ndarray,
    clients: Sequence[tuple[np.ndarray, np.ndarray]],
    rounds: int,
    local_steps: int,
    learning_rate: float,
) -> tuple[np.ndarray, dict]:
    if rounds <= 0 or local_steps <= 0:
        raise ValueError("rounds and local steps must be positive")
    parameters = initial.copy()
    sample_counts = [len(labels) for _, labels in clients]
    parameter_bytes = parameters.nbytes
    started = time.perf_counter()
    history = []
    for round_index in range(rounds):
        local_parameters = [
            local_logistic_train(parameters, features, labels, local_steps, learning_rate)
            for features, labels in clients
        ]
        parameters = weighted_average(local_parameters, sample_counts)
        history.append({"round": round_index + 1, "parameter_norm": float(np.linalg.norm(parameters))})
    elapsed = time.perf_counter() - started
    return parameters, {
        "runtime_seconds": elapsed,
        "rounds": rounds,
        "local_steps": local_steps,
        "client_parameter_updates": rounds * local_steps * len(clients),
        "examples_processed": rounds * local_steps * sum(sample_counts),
        "transmitted_parameter_bytes": rounds * len(clients) * parameter_bytes * 2,
        "communication_convention": "one full-model download plus upload per client per round",
        "history": history,
    }


def make_federated_data(
    seed: int,
    train_size: int,
    test_size: int,
    dimensions: int,
    client_sizes: Sequence[int],
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]], list[dict]]:
    if sum(client_sizes) != train_size:
        raise ValueError("client sizes must sum to train size")
    rng = np.random.default_rng(seed)
    true_parameters = rng.normal(size=dimensions)
    train_x = rng.normal(size=(train_size, dimensions))
    test_x = rng.normal(size=(test_size, dimensions))
    train_logits = train_x @ true_parameters + 0.35 * rng.normal(size=train_size)
    test_logits = test_x @ true_parameters + 0.35 * rng.normal(size=test_size)
    train_y = (train_logits > 0).astype(float)
    test_y = (test_logits > 0).astype(float)
    if split == "iid":
        order = rng.permutation(train_size)
    elif split == "non_iid":
        # Label-major ordering creates heterogeneous client label proportions.
        order = np.lexsort((train_x[:, 0], train_y))
    else:
        raise ValueError("split must be iid or non_iid")
    clients = []
    manifests = []
    start = 0
    for client_id, size in enumerate(client_sizes):
        indices = order[start : start + size]
        clients.append((train_x[indices], train_y[indices]))
        manifests.append(
            {
                "client": client_id,
                "samples": size,
                "positive_fraction": float(np.mean(train_y[indices])),
            }
        )
        start += size
    return train_x, train_y, test_x, test_y, clients, manifests


def mgda_two_task_weight(gradient_one: torch.Tensor, gradient_two: torch.Tensor) -> float:
    """Return alpha minimizing ||alpha*g1 + (1-alpha)*g2||^2 on [0, 1]."""
    g1 = gradient_one.detach().reshape(-1).double()
    g2 = gradient_two.detach().reshape(-1).double()
    difference = g1 - g2
    denominator = float(torch.dot(difference, difference))
    if denominator <= 1e-20:
        return 0.5
    alpha = float((torch.dot(g2, g2) - torch.dot(g1, g2)) / denominator)
    return min(1.0, max(0.0, alpha))


def gradnorm_weight_step(
    weights: torch.Tensor,
    base_gradient_norms: torch.Tensor,
    losses: torch.Tensor,
    initial_losses: torch.Tensor,
    learning_rate: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One differentiable GradNorm task-weight update with sum-to-two normalization."""
    current = weights.detach().clone().requires_grad_(True)
    norms = current * base_gradient_norms.detach()
    relative = (losses.detach() / initial_losses.detach()).clamp_min(1e-12)
    inverse_rate = relative / relative.mean()
    targets = norms.detach().mean() * inverse_rate.pow(alpha)
    objective = torch.abs(norms - targets).sum()
    objective.backward()
    if current.grad is None:
        raise AssertionError("GradNorm task weights did not receive a gradient")
    gradient = current.grad.detach().clone()
    with torch.no_grad():
        updated = (current - learning_rate * gradient).clamp_min(1e-4)
        updated *= 2.0 / updated.sum()
    return updated.detach(), gradient, targets.detach()


class TwoTaskNetwork(nn.Module):
    def __init__(self, dimensions: int, hidden: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(dimensions, hidden), nn.Tanh())
        self.head_one = nn.Linear(hidden, 1)
        self.head_two = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.shared(features)
        return self.head_one(representation).squeeze(-1), self.head_two(representation).squeeze(-1)


def make_multitask_data(seed: int, train_size: int, test_size: int, dimensions: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(train_size + test_size, dimensions, generator=generator)
    noise_one = 0.35 * torch.randn(train_size + test_size, generator=generator)
    noise_two = 0.55 * torch.randn(train_size + test_size, generator=generator)
    task_one = (features[:, 0] + 0.6 * features[:, 1] + noise_one > 0).float()
    task_two = (-0.8 * features[:, 0] + 0.7 * features[:, 2] + 0.4 * features[:, 3] + noise_two > 0).float()
    return {
        "train_x": features[:train_size],
        "train_y1": task_one[:train_size],
        "train_y2": task_two[:train_size],
        "test_x": features[train_size:],
        "test_y1": task_one[train_size:],
        "test_y2": task_two[train_size:],
    }


def _flatten_gradients(loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=False)
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def train_multitask(
    method: str,
    seed: int,
    data: dict[str, torch.Tensor],
    dimensions: int,
    hidden: int,
    epochs: int,
    learning_rate: float,
    gradnorm_learning_rate: float,
    gradnorm_alpha: float,
) -> dict:
    if method not in {"equal", "mgda", "gradnorm"}:
        raise ValueError("unknown multi-task method")
    torch.manual_seed(seed)
    model = TwoTaskNetwork(dimensions, hidden)
    initial_state_checksum = float(
        sum(parameter.detach().double().sum() for parameter in model.parameters())
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    task_weights = torch.ones(2)
    initial_losses = None
    history = []
    nonzero_weight_gradients = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        logits_one, logits_two = model(data["train_x"])
        losses = torch.stack(
            [criterion(logits_one, data["train_y1"]), criterion(logits_two, data["train_y2"])]
        )
        if initial_losses is None:
            initial_losses = losses.detach().clone()
        shared_parameters = list(model.shared.parameters())
        gradient_one = _flatten_gradients(losses[0], shared_parameters)
        gradient_two = _flatten_gradients(losses[1], shared_parameters)
        if method == "equal":
            task_weights = torch.ones(2)
            method_detail = {"alpha": 0.5}
        elif method == "mgda":
            alpha_value = mgda_two_task_weight(gradient_one, gradient_two)
            # Scale simplex weights to sum two, matching equal/GradNorm gradient scale.
            task_weights = torch.tensor([2.0 * alpha_value, 2.0 * (1.0 - alpha_value)])
            method_detail = {"alpha": alpha_value}
        else:
            base_norms = torch.stack([gradient_one.norm(), gradient_two.norm()])
            task_weights, weight_gradient, targets = gradnorm_weight_step(
                task_weights,
                base_norms,
                losses,
                initial_losses,
                gradnorm_learning_rate,
                gradnorm_alpha,
            )
            nonzero_weight_gradients += int(bool(torch.any(weight_gradient.abs() > 1e-12)))
            method_detail = {
                "weight_gradient": weight_gradient.tolist(),
                "targets": targets.tolist(),
            }
        optimizer.zero_grad()
        objective = (task_weights.detach() * losses).sum()
        objective.backward()
        optimizer.step()
        history.append(
            {
                "epoch": epoch + 1,
                "loss_one": float(losses[0].detach()),
                "loss_two": float(losses[1].detach()),
                "weight_one": float(task_weights[0]),
                "weight_two": float(task_weights[1]),
                "weight_sum": float(task_weights.sum()),
                **method_detail,
            }
        )
    runtime = time.perf_counter() - started
    with torch.no_grad():
        train_logits = model(data["train_x"])
        test_logits = model(data["test_x"])
        metrics = {}
        for split, logits, y1, y2 in (
            ("train", train_logits, data["train_y1"], data["train_y2"]),
            ("test", test_logits, data["test_y1"], data["test_y2"]),
        ):
            metrics[split] = {
                "loss_one": float(criterion(logits[0], y1)),
                "loss_two": float(criterion(logits[1], y2)),
                "accuracy_one": float(((logits[0] >= 0).float() == y1).float().mean()),
                "accuracy_two": float(((logits[1] >= 0).float() == y2).float().mean()),
            }
    return {
        "method": method,
        "seed": seed,
        "runtime_seconds": runtime,
        "epochs": epochs,
        "metrics": metrics,
        "final_weights": task_weights.tolist(),
        "nonzero_weight_gradient_epochs": nonzero_weight_gradients,
        "history": history,
        "initial_state_checksum": initial_state_checksum,
        "state_checksum": float(sum(parameter.detach().double().sum() for parameter in model.parameters())),
    }
