#!/usr/bin/env python3
"""Independently verify saved FedAvg and multi-task artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.distributed_multitask import (  # noqa: E402
    centralized_train,
    fedavg_train,
    gradnorm_weight_step,
    make_federated_data,
    mgda_two_task_weight,
    weighted_average,
)


def load(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run_dir = args.run_directory.resolve()
    config = load(run_dir / "config.json")
    summary = load(run_dir / "summary.json")
    raw = load(run_dir / "raw_results.json")
    records = load(run_dir / "records.json")

    fedavg = raw["fedavg"]
    client_count = len(config["client_sizes"])
    parameter_count = config["dimensions"] + 1
    byte_formula = all(
        record["transmitted_parameter_bytes"]
        == record["rounds"] * client_count * parameter_count * 8 * 2
        for record in fedavg
    )
    example_formula = all(
        record["examples_processed"]
        == record["rounds"] * record["local_steps"] * config["train_size"]
        for record in fedavg
    )
    update_formula = all(
        record["client_parameter_updates"]
        == record["rounds"] * record["local_steps"] * client_count
        for record in fedavg
    )
    same_compute_valid = all(
        record["centralized_steps"] == config["same_compute_total_steps"]
        and record["examples_processed"] == config["same_compute_total_steps"] * config["train_size"]
        for record in fedavg
        if record["view"] == "same_compute"
    )
    communication_valid = all(
        record["rounds"] == config["communication_rounds"]
        for record in fedavg
        if record["view"] == "same_communication_rounds"
    )

    non_iid_manifests = [item for item in raw["client_manifests"] if item["split"] == "non_iid"]
    iid_manifests = [item for item in raw["client_manifests"] if item["split"] == "iid"]
    non_iid_heterogeneous = all(
        min(client["positive_fraction"] for client in item["clients"]) == 0.0
        and max(client["positive_fraction"] for client in item["clients"]) == 1.0
        for item in non_iid_manifests
    )
    unequal_clients = len(set(config["client_sizes"])) > 1

    # Fresh algebraic FedAvg check: one local full-batch step must equal one centralized step.
    train_x, train_y, _, _, clients, _ = make_federated_data(
        101, 60, 20, 3, [5, 10, 15, 30], "non_iid"
    )
    initial = np.zeros(4)
    central, _ = centralized_train(initial, train_x, train_y, 1, 0.1)
    federated, accounting = fedavg_train(initial, clients, 1, 1, 0.1)
    fresh_one_step_equal = np.allclose(central, federated, rtol=1e-12, atol=1e-12)

    weighted = weighted_average([np.array([0.0]), np.array([10.0])], [1, 3])
    weighted_oracle_valid = np.allclose(weighted, [7.5])
    saved_weight_check = raw["weighted_aggregation_check"]

    multitask = raw["multitask"]
    same_initialization = all(
        len({record["initial_state_checksum"] for record in multitask if record["seed"] == seed}) == 1
        for seed in config["seeds"]
    )
    same_budget = all(record["epochs"] == config["multitask_epochs"] for record in multitask)
    metric_ranges = all(
        0 <= record["metrics"]["test"][metric] <= 1
        for record in multitask
        for metric in ("accuracy_one", "accuracy_two")
    ) and all(
        record["metrics"]["test"][metric] > 0
        for record in multitask
        for metric in ("loss_one", "loss_two")
    )
    runtimes_positive = all(record["runtime_seconds"] > 0 for record in multitask)

    known = raw["known_mgda"]
    known_mgda_valid = math.isclose(known["opposing"], 0.5) and math.isclose(
        known["orthogonal_unequal"], 0.8
    )
    alpha = mgda_two_task_weight(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0]))
    combined = alpha * torch.tensor([1.0, 0.0]) + (1 - alpha) * torch.tensor([0.0, 2.0])
    mgda_minimum_valid = alpha == 0.8 and combined.norm() <= torch.tensor([1.0, 0.0]).norm()

    gradnorm_records = [record for record in multitask if record["method"] == "gradnorm"]
    gradnorm_sum_two = all(
        abs(epoch["weight_sum"] - 2.0) < 1e-5
        for record in gradnorm_records
        for epoch in record["history"]
    )
    gradnorm_nonzero = all(record["nonzero_weight_gradient_epochs"] > 0 for record in gradnorm_records)
    updated, gradient, targets = gradnorm_weight_step(
        torch.ones(2),
        torch.tensor([2.0, 0.5]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 2.0]),
        0.05,
        1.5,
    )
    expected_gradient = torch.sign(torch.tensor([2.0, 0.5]) - targets) * torch.tensor([2.0, 0.5])
    gradnorm_gradient_valid = torch.allclose(gradient, expected_gradient) and math.isclose(
        float(updated.sum()), 2.0, rel_tol=1e-6
    )

    checks = {
        "final_configuration": config["run_kind"] == "final",
        "three_seeds": len(config["seeds"]) == 3,
        "iid_and_non_iid_present": {record["split"] for record in fedavg} == {"iid", "non_iid"},
        "unequal_client_sizes": unequal_clients,
        "non_iid_clients_are_heterogeneous": non_iid_heterogeneous,
        "iid_client_manifests_present": len(iid_manifests) == len(config["seeds"]),
        "sample_weighting_oracle": weighted_oracle_valid and saved_weight_check["different_due_to_unequal_clients"],
        "transmitted_byte_formula": byte_formula,
        "example_compute_formula": example_formula,
        "client_update_formula": update_formula,
        "same_compute_view_valid": same_compute_valid,
        "same_communication_view_valid": communication_valid,
        "local_steps_varied": {record["local_steps"] for record in fedavg} >= {1, 5, 10},
        "rounds_varied": len({record["rounds"] for record in fedavg}) >= 3,
        "fresh_one_step_fedavg_equals_centralized": bool(fresh_one_step_equal),
        "fresh_fedavg_accounting": accounting["examples_processed"] == 60,
        "multitask_methods_present": {record["method"] for record in multitask} == {"equal", "mgda", "gradnorm"},
        "same_initialization_per_seed": same_initialization,
        "same_epoch_budget": same_budget,
        "metric_ranges_valid": metric_ranges,
        "multitask_runtimes_positive": runtimes_positive,
        "known_mgda_cases": known_mgda_valid,
        "mgda_constrained_minimum": bool(mgda_minimum_valid),
        "gradnorm_weight_sums_two": gradnorm_sum_two,
        "gradnorm_nonzero_weight_gradients": gradnorm_nonzero,
        "gradnorm_gradient_formula": bool(gradnorm_gradient_valid),
        "single_machine_disclosure": "single-machine local simulation" in summary["environment"]["execution"],
        "record_count": len(records) == 45,
        "plot_present": (run_dir / "distributed_multitask_comparison.png").stat().st_size > 0,
        "status_passed": load(run_dir / "status.json")["passed"] is True,
    }
    result = {"passed": all(checks.values()), "check_count": len(checks), "checks": checks}
    (run_dir / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
