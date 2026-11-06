#!/usr/bin/env python3
"""Run Track D FedAvg and multi-task optimization experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.distributed_multitask import (  # noqa: E402
    centralized_train,
    evaluate_logistic,
    fedavg_train,
    make_federated_data,
    make_multitask_data,
    mgda_two_task_weight,
    train_multitask,
    weighted_average,
)


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = ROOT / "results" / "track_d_distributed" / run_id
    run_dir.mkdir(parents=True)
    dump(run_dir / "config.json", config)

    fedavg_records = []
    client_manifests = []
    centralized_cache = {}
    for seed in config["seeds"]:
        for split in ("iid", "non_iid"):
            train_x, train_y, test_x, test_y, clients, manifest = make_federated_data(
                seed,
                config["train_size"],
                config["test_size"],
                config["dimensions"],
                config["client_sizes"],
                split,
            )
            client_manifests.append({"seed": seed, "split": split, "clients": manifest})
            initial = np.zeros(config["dimensions"] + 1)
            schedules = []
            for local_steps in config["same_compute_local_steps"]:
                if config["same_compute_total_steps"] % local_steps:
                    raise ValueError("same-compute total steps must be divisible by local steps")
                schedules.append(
                    {
                        "view": "same_compute",
                        "rounds": config["same_compute_total_steps"] // local_steps,
                        "local_steps": local_steps,
                        "centralized_steps": config["same_compute_total_steps"],
                    }
                )
            for local_steps in config["communication_local_steps"]:
                schedules.append(
                    {
                        "view": "same_communication_rounds",
                        "rounds": config["communication_rounds"],
                        "local_steps": local_steps,
                        "centralized_steps": config["communication_rounds"] * local_steps,
                    }
                )
            for schedule in schedules:
                cache_key = (seed, schedule["centralized_steps"])
                if cache_key not in centralized_cache:
                    central_parameters, central_runtime = centralized_train(
                        initial,
                        train_x,
                        train_y,
                        schedule["centralized_steps"],
                        config["fedavg_learning_rate"],
                    )
                    centralized_cache[cache_key] = {
                        "metrics": evaluate_logistic(central_parameters, test_x, test_y),
                        "runtime_seconds": central_runtime,
                        "parameters": central_parameters.tolist(),
                    }
                fed_parameters, accounting = fedavg_train(
                    initial,
                    clients,
                    schedule["rounds"],
                    schedule["local_steps"],
                    config["fedavg_learning_rate"],
                )
                metrics = evaluate_logistic(fed_parameters, test_x, test_y)
                central = centralized_cache[cache_key]
                fedavg_records.append(
                    {
                        "seed": seed,
                        "split": split,
                        **schedule,
                        "test_loss": metrics["loss"],
                        "test_accuracy": metrics["accuracy"],
                        "centralized_test_loss": central["metrics"]["loss"],
                        "centralized_test_accuracy": central["metrics"]["accuracy"],
                        "accuracy_gap_vs_centralized": metrics["accuracy"] - central["metrics"]["accuracy"],
                        "runtime_seconds": accounting["runtime_seconds"],
                        "centralized_runtime_seconds": central["runtime_seconds"],
                        "client_parameter_updates": accounting["client_parameter_updates"],
                        "examples_processed": accounting["examples_processed"],
                        "transmitted_parameter_bytes": accounting["transmitted_parameter_bytes"],
                        "parameter_count": len(initial),
                        "parameter_checksum": float(np.sum(fed_parameters)),
                        "history": accounting["history"],
                        "communication_convention": accounting["communication_convention"],
                    }
                )

    # Directly demonstrate why sample-count weighting matters on unequal clients.
    weighted_oracle = weighted_average(
        [np.full(2, float(index)) for index in range(len(config["client_sizes"]))],
        config["client_sizes"],
    ).tolist()
    unweighted_reference = np.mean(
        [np.full(2, float(index)) for index in range(len(config["client_sizes"]))], axis=0
    ).tolist()

    multitask_data = make_multitask_data(
        seed=20261028,
        train_size=config["multitask_train_size"],
        test_size=config["multitask_test_size"],
        dimensions=config["dimensions"],
    )
    multitask_records = []
    for seed in config["seeds"]:
        for method in ("equal", "mgda", "gradnorm"):
            result = train_multitask(
                method,
                seed,
                multitask_data,
                config["dimensions"],
                config["multitask_hidden"],
                config["multitask_epochs"],
                config["multitask_learning_rate"],
                config["gradnorm_learning_rate"],
                config["gradnorm_alpha"],
            )
            multitask_records.append(result)

    known_mgda = {
        "opposing": mgda_two_task_weight(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])),
        "orthogonal_unequal": mgda_two_task_weight(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0])),
        "expected_opposing": 0.5,
        "expected_orthogonal_unequal": 0.8,
    }

    fedavg_summary = {}
    for split in ("iid", "non_iid"):
        fedavg_summary[split] = {}
        for view in ("same_compute", "same_communication_rounds"):
            matching = [record for record in fedavg_records if record["split"] == split and record["view"] == view]
            fedavg_summary[split][view] = []
            for local_steps in sorted({record["local_steps"] for record in matching}):
                group = [record for record in matching if record["local_steps"] == local_steps]
                fedavg_summary[split][view].append(
                    {
                        "local_steps": local_steps,
                        "rounds": group[0]["rounds"],
                        "centralized_steps": group[0]["centralized_steps"],
                        "accuracy": summarize([record["test_accuracy"] for record in group]),
                        "centralized_accuracy": summarize([record["centralized_test_accuracy"] for record in group]),
                        "accuracy_gap": summarize([record["accuracy_gap_vs_centralized"] for record in group]),
                        "runtime_seconds": summarize([record["runtime_seconds"] for record in group]),
                        "transmitted_parameter_bytes": group[0]["transmitted_parameter_bytes"],
                        "examples_processed": group[0]["examples_processed"],
                    }
                )

    multitask_summary = {}
    for method in ("equal", "mgda", "gradnorm"):
        group = [record for record in multitask_records if record["method"] == method]
        multitask_summary[method] = {
            metric: summarize([record["metrics"]["test"][metric] for record in group])
            for metric in ("loss_one", "loss_two", "accuracy_one", "accuracy_two")
        }
        multitask_summary[method]["runtime_seconds"] = summarize(
            [record["runtime_seconds"] for record in group]
        )
        multitask_summary[method]["final_weight_one"] = summarize(
            [record["final_weights"][0] for record in group]
        )
        multitask_summary[method]["final_weight_two"] = summarize(
            [record["final_weights"][1] for record in group]
        )

    initial_checksums_equal = all(
        len(
            {
                record["initial_state_checksum"]
                for record in multitask_records
                if record["seed"] == seed
            }
        )
        == 1
        for seed in config["seeds"]
    )
    gradnorm_records = [record for record in multitask_records if record["method"] == "gradnorm"]
    gradnorm_checks = {
        "all_weight_sums_two": all(
            abs(epoch["weight_sum"] - 2.0) < 1e-5
            for record in gradnorm_records
            for epoch in record["history"]
        ),
        "weights_received_nonzero_gradients": all(
            record["nonzero_weight_gradient_epochs"] > 0 for record in gradnorm_records
        ),
        "weights_positive": all(
            min(record["final_weights"]) > 0 for record in gradnorm_records
        ),
    }

    raw = {
        "fedavg": fedavg_records,
        "client_manifests": client_manifests,
        "centralized_cache": {
            f"seed={seed},steps={steps}": value for (seed, steps), value in centralized_cache.items()
        },
        "weighted_aggregation_check": {
            "weighted": weighted_oracle,
            "unweighted": unweighted_reference,
            "different_due_to_unequal_clients": weighted_oracle != unweighted_reference,
        },
        "multitask": multitask_records,
        "known_mgda": known_mgda,
    }
    dump(run_dir / "raw_results.json", raw)

    flat_records = []
    for record in fedavg_records:
        flat_records.append({key: value for key, value in record.items() if key != "history"} | {"experiment": "fedavg"})
    for record in multitask_records:
        flat_records.append(
            {
                "experiment": "multitask",
                "method": record["method"],
                "seed": record["seed"],
                "runtime_seconds": record["runtime_seconds"],
                **record["metrics"]["test"],
                "final_weight_one": record["final_weights"][0],
                "final_weight_two": record["final_weights"][1],
            }
        )
    dump(run_dir / "records.json", flat_records)
    fields = sorted({key for record in flat_records for key in record})
    with (run_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_records)

    summary = {
        "run_kind": config["run_kind"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "execution": "single-machine local simulation; not geographically distributed hardware",
        },
        "fedavg": fedavg_summary,
        "fedavg_accounting": {
            "clients": len(config["client_sizes"]),
            "client_sizes": config["client_sizes"],
            "parameter_count": config["dimensions"] + 1,
            "bytes_convention": "64-bit parameters, client download plus upload each round",
            "weighted_aggregation_check": raw["weighted_aggregation_check"],
        },
        "multitask": multitask_summary,
        "multitask_correctness": {
            "same_initialization_per_seed": initial_checksums_equal,
            "mgda_known_case": known_mgda,
            "gradnorm": gradnorm_checks,
            "same_epochs": config["multitask_epochs"],
            "same_data_split": True,
        },
    }
    dump(run_dir / "summary.json", summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for split, marker in (("iid", "o"), ("non_iid", "s")):
        points = fedavg_summary[split]["same_compute"]
        axes[0].plot(
            [point["transmitted_parameter_bytes"] for point in points],
            [point["accuracy"]["mean"] for point in points],
            marker=marker,
            label=split.replace("_", "-"),
        )
    axes[0].set_xlabel("Transmitted parameter bytes")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("FedAvg: same compute, communication trade-off")
    axes[0].legend()
    methods = ["equal", "mgda", "gradnorm"]
    x = np.arange(len(methods))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        [multitask_summary[method]["accuracy_one"]["mean"] for method in methods],
        width,
        label="task 1",
    )
    axes[1].bar(
        x + width / 2,
        [multitask_summary[method]["accuracy_two"]["mean"] for method in methods],
        width,
        label="task 2",
    )
    axes[1].set_xticks(x, methods)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Multi-task trade-offs")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "distributed_multitask_comparison.png", dpi=160)
    plt.close(fig)

    dump(run_dir / "status.json", {"passed": True, "run_directory": run_dir.relative_to(ROOT).as_posix()})
    dump(
        ROOT / "results" / "track_d_distributed" / "latest-success.json",
        {
            "passed": True,
            "run_kind": config["run_kind"],
            "run_directory": run_dir.relative_to(ROOT).as_posix(),
            "records": len(flat_records),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
