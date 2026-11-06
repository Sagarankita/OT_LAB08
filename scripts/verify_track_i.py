#!/usr/bin/env python3
"""Independently verify the latest successful Track I experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pennylane as qml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.quantum_algorithms import (  # noqa: E402
    apply_bernstein_vazirani,
    apply_deutsch_jozsa,
    apply_grover,
    apply_quantum_annealing,
)


def main() -> int:
    latest = json.loads((ROOT / "results/track_i_quantum/latest-success.json").read_text())
    run_dir = ROOT / latest["run_directory"]
    raw = json.loads((run_dir / "raw_results.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    algorithms, qml_results = raw["algorithms"], raw["qml"]
    checks: list[tuple[str, bool]] = []

    def check(name: str, condition: bool) -> None:
        checks.append((name, bool(condition)))

    check("latest run passed", latest["passed"])
    check("final configuration", config["run_kind"] == "final" and len(config["seeds"]) == 3)
    check("simulator disclosure", "no real quantum hardware" in summary["environment"]["execution"])
    dj = algorithms["deutsch_jozsa"]
    check("DJ constant known answer", dj["constant_classified"] and set(dj["constant_counts"]) == {"000"})
    check("DJ balanced known answer", dj["balanced_classified"] and "000" not in dj["balanced_counts"])
    check("DJ one query", dj["oracle_queries"] == 1)
    bv = algorithms["bernstein_vazirani"]
    check("BV secret recovered", bv["recovered_secret"] == "1011")
    check("BV query reduction", bv["oracle_queries"] == 1 < bv["classical_deterministic_queries"])
    grover = algorithms["grover"]
    check("Grover success", grover["success_probability"] > 0.9)
    check("Grover query reduction fixed case", grover["iterations_and_oracle_queries"] < grover["classical_queries_for_fixed_target"])
    check("Grover query reduction worst case", grover["iterations_and_oracle_queries"] < grover["classical_worst_case_queries"])
    check("simulator has overhead", grover["quantum_simulator_wall_seconds"]["mean"] > grover["classical_wall_seconds"]["mean"])
    check("QFT analytic fidelity", algorithms["qft"]["analytic_state_fidelity"] > 1 - 1e-10)
    check("inverse QFT round trip", algorithms["qft"]["roundtrip_probability"] > 1 - 1e-10)
    check("QPE phase recovered", abs(algorithms["qpe"]["estimated_phase"] - 3 / 8) < 1e-12)
    check("QPE deterministic", algorithms["qpe"]["success_probability"] > 1 - 1e-10)
    shor = algorithms["shor_compiled_15"]
    check("Shor factors 15", shor["factors"] == [3, 5])
    check("Shor inferred order", shor["inferred_orders"] == [4])
    check("Shor limitation disclosed", "not general" in shor["disclosure"])
    annealing = algorithms["quantum_annealing"]
    check("annealing simulator disclosed", "simulation" in annealing["hardware"])
    check("annealing probabilities normalized", all(abs(run["probability_sum"] - 1) < 1e-9 for run in annealing["runs"]))
    check("slower anneal improves success", annealing["runs"][-1]["ground_success_probability"] > annealing["runs"][0]["ground_success_probability"])
    check("slow anneal high success", annealing["runs"][-1]["ground_success_probability"] > 0.9)
    check("annealer embedding limit stated", "chains" in annealing["embedding_note"])

    dataset = qml_results["dataset"]
    split = dataset["train"] + dataset["validation"] + dataset["test"]
    check("same disjoint QML split", len(split) == len(set(split)) == dataset["samples"])
    check("preprocessing train-only", "training split only" in dataset["preprocessing"])
    vqc = qml_results["variational_quantum_classifier"]
    mlp = qml_results["classical_mlp"]
    check("VQC repeated seeds", [item["seed"] for item in vqc] == config["seeds"])
    check("classical repeated seeds", [item["seed"] for item in mlp] == config["seeds"])
    check("VQC circuit cost recorded", all(item["circuit_executions"] > 0 for item in vqc))
    check("classical circuit count zero", all(item["circuit_executions"] == 0 for item in mlp))
    check("VQC metric range", all(0 <= item["accuracies"]["test"] <= 1 for item in vqc))
    check("matched epochs", all(item["epochs"] == config["qml_epochs"] for item in vqc + mlp))
    noisy = qml_results["noisy_quantum_classifier"]
    check("finite-shot noisy execution", all(item["shots_per_prediction"] == config["noise_shots"] for item in noisy))
    check("noise model disclosed", all("depolarizing" in item["noise"] for item in noisy))
    qkernel = qml_results["quantum_kernel"]
    check("quantum kernel circuit count", qkernel["circuit_executions"] == 684)
    check("quantum kernel numerically PSD", qkernel["train_kernel_min_eigenvalue"] > -1e-10)
    check("result plot exists", (run_dir / "quantum_algorithms_qml.png").stat().st_size > 1000)

    # Fresh executions make the verifier independent of saved labels.
    dev = qml.device("default.qubit", wires=5)

    @qml.qnode(dev)
    def fresh_dj():
        apply_deutsch_jozsa([0, 1, 2], 3, True)
        return qml.probs(wires=[0, 1, 2])

    @qml.qnode(dev)
    def fresh_bv():
        apply_bernstein_vazirani([1, 0, 1, 1], [0, 1, 2, 3], 4)
        return qml.probs(wires=[0, 1, 2, 3])

    @qml.qnode(qml.device("default.qubit", wires=3))
    def fresh_grover():
        apply_grover([1, 0, 1], [0, 1, 2], 2)
        return qml.probs(wires=[0, 1, 2])

    @qml.qnode(qml.device("default.qubit", wires=3))
    def fresh_anneal():
        apply_quantum_annealing(3, 2.0, 20)
        return qml.probs(wires=[0, 1, 2])

    check("fresh DJ balanced", fresh_dj()[0] < 1e-10)
    check("fresh BV secret", int(np.argmax(fresh_bv())) == 11)
    check("fresh Grover success", fresh_grover()[5] > 0.9)
    check("fresh annealing unit norm", abs(float(np.sum(fresh_anneal())) - 1) < 1e-10)

    failed = [name for name, passed in checks if not passed]
    report = {
        "passed": not failed,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed": failed,
        "verified_run": latest["run_directory"],
    }
    (run_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
