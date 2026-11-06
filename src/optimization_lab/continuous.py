"""Continuous-optimization experiments and independent mathematical checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares


FIRST_ORDER_METHODS = (
    "gd", "sgd", "mini_batch", "momentum", "nag", "adagrad", "rmsprop", "adam"
)


@dataclass(frozen=True)
class OptimizationResult:
    parameters: np.ndarray
    history: tuple[dict[str, float | int], ...]


def make_ridge_data(seed: int, samples: int, features: int,
                    observation_noise: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples <= 0 or features <= 0 or observation_noise < 0:
        raise ValueError("invalid regression dimensions or noise")
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(samples, features))
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    true_parameters = generator.normal(size=features)
    targets = matrix @ true_parameters + generator.normal(scale=observation_noise, size=samples)
    return matrix, targets, true_parameters


def ridge_loss(parameters: np.ndarray, matrix: np.ndarray, targets: np.ndarray,
               regularization: float) -> float:
    residual = matrix @ parameters - targets
    return float(0.5 * np.mean(residual * residual) +
                 0.5 * regularization * np.dot(parameters, parameters))


def ridge_gradient(parameters: np.ndarray, matrix: np.ndarray, targets: np.ndarray,
                   regularization: float) -> np.ndarray:
    return matrix.T @ (matrix @ parameters - targets) / len(targets) + regularization * parameters


def ridge_hessian(matrix: np.ndarray, regularization: float) -> np.ndarray:
    return matrix.T @ matrix / len(matrix) + regularization * np.eye(matrix.shape[1])


def ridge_optimum(matrix: np.ndarray, targets: np.ndarray, regularization: float) -> np.ndarray:
    return np.linalg.solve(ridge_hessian(matrix, regularization), matrix.T @ targets / len(targets))


def finite_difference_gradient(function, parameters: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    result = np.empty_like(parameters, dtype=float)
    for index in range(len(parameters)):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        result[index] = (function(plus) - function(minus)) / (2 * epsilon)
    return result


def optimize_ridge(method: str, matrix: np.ndarray, targets: np.ndarray,
                   regularization: float, initial: np.ndarray, epochs: int,
                   learning_rate: float, batch_size: int, seed: int,
                   gradient_noise_std: float = 0.0) -> OptimizationResult:
    if method not in FIRST_ORDER_METHODS:
        raise ValueError(f"unknown method: {method}")
    if epochs <= 0 or learning_rate <= 0 or batch_size <= 0 or gradient_noise_std < 0:
        raise ValueError("invalid optimizer settings")
    if matrix.ndim != 2 or targets.shape != (len(matrix),) or initial.shape != (matrix.shape[1],):
        raise ValueError("ridge shapes do not agree")
    generator = np.random.default_rng(seed)
    parameters = initial.astype(float, copy=True)
    velocity = np.zeros_like(parameters)
    accumulator = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    first_moment = np.zeros_like(parameters)
    evaluations = 0
    steps = 0
    optimum = ridge_optimum(matrix, targets, regularization)
    optimum_loss = ridge_loss(optimum, matrix, targets, regularization)
    history: list[dict[str, float | int]] = []

    def record(epoch: int) -> None:
        loss = ridge_loss(parameters, matrix, targets, regularization)
        history.append({
            "epoch": epoch, "loss": loss, "objective_gap": max(0.0, loss - optimum_loss),
            "gradient_norm": float(np.linalg.norm(
                ridge_gradient(parameters, matrix, targets, regularization)
            )),
            "example_gradient_evaluations": evaluations, "parameter_updates": steps,
        })

    record(0)
    n = len(matrix)
    effective_batch = n if method == "gd" else 1 if method == "sgd" else min(batch_size, n)
    beta1, beta2, momentum, epsilon = 0.9, 0.999, 0.9, 1e-8
    for epoch in range(1, epochs + 1):
        order = generator.permutation(n) if effective_batch < n else np.arange(n)
        for start in range(0, n, effective_batch):
            batch = order[start:start + effective_batch]
            batch_matrix, batch_targets = matrix[batch], targets[batch]
            if method == "nag":
                lookahead = parameters + momentum * velocity
                gradient = ridge_gradient(lookahead, batch_matrix, batch_targets, regularization)
            else:
                gradient = ridge_gradient(parameters, batch_matrix, batch_targets, regularization)
            if gradient_noise_std:
                gradient = gradient + generator.normal(
                    scale=gradient_noise_std / math.sqrt(len(batch)), size=gradient.shape
                )
            evaluations += len(batch)
            steps += 1
            if method in {"gd", "sgd", "mini_batch"}:
                parameters -= learning_rate * gradient
            elif method == "momentum":
                velocity = momentum * velocity - learning_rate * gradient
                parameters += velocity
            elif method == "nag":
                velocity = momentum * velocity - learning_rate * gradient
                parameters += velocity
            elif method == "adagrad":
                accumulator += gradient * gradient
                parameters -= learning_rate * gradient / (np.sqrt(accumulator) + epsilon)
            elif method == "rmsprop":
                second_moment = 0.99 * second_moment + 0.01 * gradient * gradient
                parameters -= learning_rate * gradient / (np.sqrt(second_moment) + epsilon)
            elif method == "adam":
                first_moment = beta1 * first_moment + (1 - beta1) * gradient
                second_moment = beta2 * second_moment + (1 - beta2) * gradient * gradient
                corrected_first = first_moment / (1 - beta1 ** steps)
                corrected_second = second_moment / (1 - beta2 ** steps)
                parameters -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
        if not np.all(np.isfinite(parameters)):
            raise FloatingPointError(f"{method} produced non-finite parameters")
        record(epoch)
    return OptimizationResult(parameters, tuple(history))


def quadratic_gradient_descent(diagonal: Sequence[float], initial: Sequence[float],
                               learning_rate: float, iterations: int) -> list[dict[str, Any]]:
    diagonal_array = np.asarray(diagonal, dtype=float)
    point = np.asarray(initial, dtype=float).copy()
    if diagonal_array.shape != point.shape or np.any(diagonal_array < 0):
        raise ValueError("quadratic diagonal and initial point are invalid")
    trajectory = []
    for iteration in range(iterations + 1):
        trajectory.append({
            "iteration": iteration, "point": point.tolist(),
            "objective": float(0.5 * np.dot(diagonal_array * point, point)),
            "gradient_norm": float(np.linalg.norm(diagonal_array * point)),
        })
        point -= learning_rate * diagonal_array * point
    return trajectory


def newton_quadratic(hessian: np.ndarray, linear: np.ndarray,
                     initial: np.ndarray, iterations: int = 3) -> OptimizationResult:
    if hessian.shape != (len(linear), len(linear)) or initial.shape != linear.shape:
        raise ValueError("quadratic dimensions do not agree")
    if np.linalg.eigvalsh(hessian).min() <= 0:
        raise ValueError("Newton demonstration requires a positive-definite Hessian")
    point = initial.astype(float, copy=True)
    history = []
    optimum = np.linalg.solve(hessian, linear)
    for iteration in range(iterations + 1):
        gradient = hessian @ point - linear
        objective = 0.5 * point @ hessian @ point - linear @ point
        history.append({
            "iteration": iteration, "objective": float(objective),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "distance_to_optimum": float(np.linalg.norm(point - optimum)),
        })
        if iteration < iterations:
            point -= np.linalg.solve(hessian, gradient)
    return OptimizationResult(point, tuple(history))


def exponential_residuals(parameters: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b = parameters
    return a * np.exp(b * x) - y


def exponential_jacobian(parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b = parameters
    exponential = np.exp(b * x)
    return np.column_stack((exponential, a * x * exponential))


def gauss_newton_exponential(x: np.ndarray, y: np.ndarray, initial: np.ndarray,
                             iterations: int = 20, damping: float = 1e-8) -> OptimizationResult:
    point = initial.astype(float, copy=True)
    history = []
    for iteration in range(iterations + 1):
        residual = exponential_residuals(point, x, y)
        jacobian = exponential_jacobian(point, x)
        gradient = jacobian.T @ residual
        objective = 0.5 * np.dot(residual, residual)
        history.append({
            "iteration": iteration, "objective": float(objective),
            "gradient_norm": float(np.linalg.norm(gradient)),
        })
        if iteration < iterations:
            step = np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(2), gradient)
            point -= step
            if not np.all(np.isfinite(point)):
                raise FloatingPointError("Gauss-Newton produced non-finite parameters")
    return OptimizationResult(point, tuple(history))


def verify_gauss_newton(x: np.ndarray, y: np.ndarray, initial: np.ndarray,
                        result: OptimizationResult) -> dict[str, Any]:
    reference = least_squares(lambda p: exponential_residuals(p, x, y), initial,
                              jac=lambda p: exponential_jacobian(p, x), method="trf")
    return {
        "scipy_success": bool(reference.success),
        "scipy_parameters": reference.x.tolist(),
        "parameter_distance": float(np.linalg.norm(result.parameters - reference.x)),
        "objective_difference": abs(
            result.history[-1]["objective"] - 0.5 * float(np.dot(reference.fun, reference.fun))
        ),
    }


def nonconvex_stationary_points() -> list[dict[str, float | str]]:
    # f(x)=x^4-x^2+0.1x has two unequal local minima and one local maximum.
    roots = np.roots([4.0, 0.0, -2.0, 0.1])
    points = []
    for root in sorted(float(value.real) for value in roots if abs(value.imag) < 1e-10):
        curvature = 12 * root * root - 2
        kind = "local_minimum" if curvature > 0 else "local_maximum"
        points.append({
            "x": root, "objective": root ** 4 - root ** 2 + 0.1 * root,
            "curvature": curvature, "classification": kind,
        })
    minimum_value = min(point["objective"] for point in points if point["classification"] == "local_minimum")
    for point in points:
        if point["classification"] == "local_minimum":
            point["classification"] = (
                "global_minimum" if math.isclose(point["objective"], minimum_value, abs_tol=1e-12)
                else "nonglobal_local_minimum"
            )
    return points


def descend_nonconvex(initial: float, learning_rate: float = 0.05,
                      iterations: int = 500) -> dict[str, float]:
    point = float(initial)
    for _ in range(iterations):
        point -= learning_rate * (4 * point ** 3 - 2 * point + 0.1)
    return {"initial": initial, "final_x": point,
            "final_objective": point ** 4 - point ** 2 + 0.1 * point}


def pack_network(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: float) -> np.ndarray:
    return np.concatenate((w1.ravel(), b1, w2, np.array([b2], dtype=float)))


def unpack_network(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if parameters.shape != (13,):
        raise ValueError("XOR network must have exactly 13 parameters")
    return parameters[:6].reshape(3, 2), parameters[6:9], parameters[9:12], float(parameters[12])


def xor_loss_and_backprop(parameters: np.ndarray, inputs: np.ndarray, labels: np.ndarray,
                          regularization: float = 1e-3) -> tuple[float, np.ndarray, np.ndarray]:
    w1, b1, w2, b2 = unpack_network(parameters)
    hidden = np.tanh(inputs @ w1.T + b1)
    logits = hidden @ w2 + b2
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    epsilon = 1e-12
    loss = -np.mean(labels * np.log(probabilities + epsilon) +
                    (1 - labels) * np.log(1 - probabilities + epsilon))
    loss += 0.5 * regularization * (np.sum(w1 * w1) + np.sum(w2 * w2))
    dlogits = (probabilities - labels) / len(labels)
    grad_w2 = hidden.T @ dlogits + regularization * w2
    grad_b2 = float(np.sum(dlogits))
    dhidden = dlogits[:, None] * w2[None, :]
    dz1 = dhidden * (1 - hidden * hidden)
    grad_w1 = dz1.T @ inputs + regularization * w1
    grad_b1 = dz1.sum(axis=0)
    gradient = pack_network(grad_w1, grad_b1, grad_w2, grad_b2)
    return float(loss), gradient, probabilities


def train_xor_adam(initial: np.ndarray, iterations: int = 1500,
                   learning_rate: float = 0.03) -> OptimizationResult:
    inputs = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    labels = np.array([0., 1., 1., 0.])
    parameters = initial.copy()
    first = np.zeros_like(parameters)
    second = np.zeros_like(parameters)
    history = []
    for step in range(1, iterations + 1):
        loss, gradient, probabilities = xor_loss_and_backprop(parameters, inputs, labels)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        parameters -= learning_rate * (first / (1 - 0.9 ** step)) / (
            np.sqrt(second / (1 - 0.999 ** step)) + 1e-8
        )
        if step == 1 or step % 100 == 0 or step == iterations:
            accuracy = float(np.mean((probabilities >= 0.5) == labels))
            history.append({"iteration": step, "loss": loss,
                            "gradient_norm": float(np.linalg.norm(gradient)),
                            "accuracy": accuracy})
    return OptimizationResult(parameters, tuple(history))


def logistic_loss_gradient_hessian(parameters: np.ndarray, matrix: np.ndarray,
                                   labels: np.ndarray, regularization: float):
    logits = matrix @ parameters
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    loss = float(np.mean(np.logaddexp(0.0, logits) - labels * logits) +
                 0.5 * regularization * np.dot(parameters, parameters))
    gradient = matrix.T @ (probabilities - labels) / len(labels) + regularization * parameters
    weights = probabilities * (1 - probabilities)
    hessian = matrix.T @ (weights[:, None] * matrix) / len(labels) + regularization * np.eye(matrix.shape[1])
    per_example_gradients = (probabilities - labels)[:, None] * matrix
    empirical_fisher = per_example_gradients.T @ per_example_gradients / len(labels)
    expected_fisher = matrix.T @ (weights[:, None] * matrix) / len(labels)
    return loss, gradient, hessian, empirical_fisher, expected_fisher


def curvature_diagnostics(parameters: np.ndarray, matrix: np.ndarray, labels: np.ndarray,
                          regularization: float, direction: np.ndarray,
                          rank: int = 2) -> dict[str, Any]:
    _, gradient, hessian, empirical_fisher, expected_fisher = logistic_loss_gradient_hessian(
        parameters, matrix, labels, regularization
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    order = np.argsort(eigenvalues)[::-1][:rank]
    low_rank = (eigenvectors[:, order] * eigenvalues[order]) @ eigenvectors[:, order].T
    diagonal = np.diag(np.diag(hessian))
    epsilon = 1e-5
    plus_gradient = logistic_loss_gradient_hessian(
        parameters + epsilon * direction, matrix, labels, regularization
    )[1]
    minus_gradient = logistic_loss_gradient_hessian(
        parameters - epsilon * direction, matrix, labels, regularization
    )[1]
    finite_hvp = (plus_gradient - minus_gradient) / (2 * epsilon)
    exact_hvp = hessian @ direction
    norm = np.linalg.norm(hessian)
    return {
        "hessian": hessian.tolist(), "diagonal": diagonal.tolist(),
        "low_rank": low_rank.tolist(), "rank": rank,
        "empirical_fisher": empirical_fisher.tolist(),
        "expected_fisher_plus_regularization": (
            expected_fisher + regularization * np.eye(matrix.shape[1])
        ).tolist(),
        "expected_fisher_relative_error": float(np.linalg.norm(
            hessian - expected_fisher - regularization * np.eye(matrix.shape[1])
        ) / norm),
        "empirical_fisher_relative_error": float(np.linalg.norm(
            hessian - empirical_fisher - regularization * np.eye(matrix.shape[1])
        ) / norm),
        "diagonal_relative_error": float(np.linalg.norm(hessian - diagonal) / norm),
        "low_rank_relative_error": float(np.linalg.norm(hessian - low_rank) / norm),
        "exact_hvp": exact_hvp.tolist(), "finite_difference_hvp": finite_hvp.tolist(),
        "hvp_relative_error": float(np.linalg.norm(exact_hvp - finite_hvp) /
                                    max(1e-15, np.linalg.norm(exact_hvp))),
        "gradient_norm": float(np.linalg.norm(gradient)),
    }


def noisy_quadratic(seed: int, strategy: str, iterations: int = 200,
                    noise_std: float = 0.5) -> dict[str, float | int | str]:
    if strategy not in {"constant_step", "decaying_step"}:
        raise ValueError("unknown noisy-objective strategy")
    generator = np.random.default_rng(seed)
    point = -4.0
    for iteration in range(1, iterations + 1):
        learning_rate = 0.05 if strategy == "constant_step" else 0.25 / math.sqrt(iteration)
        noisy_gradient = (point - 3.0) + generator.normal(scale=noise_std)
        point -= learning_rate * noisy_gradient
    return {
        "seed": seed, "strategy": strategy, "iterations": iterations,
        "final_x": point, "absolute_error": abs(point - 3.0),
        "final_objective": 0.5 * (point - 3.0) ** 2,
    }

