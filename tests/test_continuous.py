import math

import numpy as np

from optimization_lab.continuous import (
    FIRST_ORDER_METHODS, curvature_diagnostics, descend_nonconvex,
    exponential_jacobian, exponential_residuals, finite_difference_gradient,
    gauss_newton_exponential, logistic_loss_gradient_hessian, make_ridge_data,
    newton_quadratic, nonconvex_stationary_points, optimize_ridge,
    pack_network, quadratic_gradient_descent, ridge_gradient, ridge_loss,
    ridge_optimum, train_xor_adam, unpack_network, verify_gauss_newton,
    xor_loss_and_backprop,
)


def test_ridge_gradient_and_known_optimum():
    x, y, _ = make_ridge_data(2, 40, 4)
    point = np.array([0.2, -0.1, 0.3, 0.4])
    analytic = ridge_gradient(point, x, y, 0.1)
    numeric = finite_difference_gradient(lambda p: ridge_loss(p, x, y, 0.1), point)
    assert np.linalg.norm(analytic - numeric) < 1e-7
    assert np.linalg.norm(ridge_gradient(ridge_optimum(x, y, 0.1), x, y, 0.1)) < 1e-10


def test_all_first_order_methods_reduce_loss_with_equal_data_exposure():
    x, y, _ = make_ridge_data(3, 64, 5)
    initial = np.ones(5)
    rates = {"gd": .1, "sgd": .01, "mini_batch": .05, "momentum": .03,
             "nag": .03, "adagrad": .15, "rmsprop": .01, "adam": .03}
    for method in FIRST_ORDER_METHODS:
        result = optimize_ridge(method, x, y, .05, initial, 20, rates[method], 16, 7)
        assert result.history[-1]["loss"] < result.history[0]["loss"]
        assert result.history[-1]["example_gradient_evaluations"] == 64 * 20


def test_newton_reaches_quadratic_optimum_in_one_update():
    hessian = np.diag([1.0, 4.0, 9.0])
    linear = np.array([1.0, -2.0, 3.0])
    result = newton_quadratic(hessian, linear, np.array([3.0, 3.0, 3.0]), 1)
    assert np.linalg.norm(result.parameters - np.linalg.solve(hessian, linear)) < 1e-12


def test_gauss_newton_matches_independent_scipy_solution():
    x = np.linspace(0, 1, 20)
    y = 1.5 * np.exp(0.7 * x)
    initial = np.array([1.0, 0.2])
    result = gauss_newton_exponential(x, y, initial)
    check = verify_gauss_newton(x, y, initial, result)
    assert check["scipy_success"]
    assert check["parameter_distance"] < 1e-7


def test_manual_backprop_matches_finite_difference_and_trains_xor():
    rng = np.random.default_rng(5)
    parameters = rng.normal(scale=.3, size=13)
    inputs = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    labels = np.array([0., 1., 1., 0.])
    _, gradient, _ = xor_loss_and_backprop(parameters, inputs, labels)
    numeric = finite_difference_gradient(
        lambda p: xor_loss_and_backprop(p, inputs, labels)[0], parameters
    )
    assert np.linalg.norm(gradient - numeric) / np.linalg.norm(numeric) < 1e-5
    trained = train_xor_adam(parameters)
    assert trained.history[-1]["accuracy"] == 1.0


def test_curvature_hvp_and_expected_fisher():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(50, 4))
    y = (rng.random(50) > .5).astype(float)
    diagnostics = curvature_diagnostics(np.zeros(4), x, y, .1, np.ones(4))
    assert diagnostics["hvp_relative_error"] < 1e-8
    assert diagnostics["expected_fisher_relative_error"] < 1e-12


def test_landscape_examples():
    convex = quadratic_gradient_descent([1, 0], [2, 3], .5, 20)
    strong = quadratic_gradient_descent([1, 10], [2, 3], .1, 100)
    assert convex[-1]["objective"] < convex[0]["objective"]
    assert convex[-1]["point"][1] == 3
    assert strong[-1]["objective"] < 1e-8
    points = nonconvex_stationary_points()
    assert {p["classification"] for p in points} == {
        "global_minimum", "nonglobal_local_minimum", "local_maximum"
    }
    left = descend_nonconvex(-1.0)
    right = descend_nonconvex(1.0)
    assert not math.isclose(left["final_objective"], right["final_objective"], abs_tol=1e-4)
