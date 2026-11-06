"""Generate Track G aggregate metrics and plot from saved raw guest records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean, stdev

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(records, policy, workload, role, metric):
    values = [record[metric] for record in records if record["policy"] == policy and
              record["workload"] == workload and record["role"] == role]
    return {"values": values, "mean": fmean(values),
            "sample_std": stdev(values) if len(values) > 1 else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    records = json.loads((run / "records.json").read_text())
    cases = {
        "cpu_weight_1_elapsed": ("cpu", 0, "elapsed"),
        "cpu_weight_3_elapsed": ("cpu", 1, "elapsed"),
        "cpu_weight_6_elapsed": ("cpu", 2, "elapsed"),
        "mixed_background_elapsed": ("mixed", 0, "elapsed"),
        "mixed_interactive_elapsed": ("mixed", 1, "elapsed"),
        "latency_target_response": ("latency", 2, "response"),
        "latency_target_elapsed": ("latency", 2, "elapsed"),
    }
    metrics = {}
    for name, (workload, role, metric) in cases.items():
        metrics[name] = {policy: stats(records, policy, workload, role, metric)
                         for policy in ("baseline", "stride")}
        baseline = metrics[name]["baseline"]["mean"]
        stride = metrics[name]["stride"]["mean"]
        metrics[name]["change_percent"] = (100.0 * (stride - baseline) / baseline
                                             if baseline else None)
    stride_cpu_ticks = {
        str(role): stats(records, "stride", "cpu", role, "ticks")
        for role in range(3)
    }
    analysis = {
        "metric_units": "xv6 timer ticks (~0.1 s each)",
        "lower_elapsed_and_response_are_better": True,
        "metrics": metrics, "stride_cpu_run_ticks": stride_cpu_ticks,
        "interpretation": (
            "Weight-6 CPU completion improved while weight-1 completion worsened, but the "
            "mixed workload tied and latency response regressed. Tick granularity and short "
            "fixed-work tasks do not establish long-run proportional-share accuracy."
        ),
    }
    analysis_path = run / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")

    panels = [
        ("CPU completion", [("w=1", "cpu", 0, "elapsed"),
                            ("w=3", "cpu", 1, "elapsed"),
                            ("w=6", "cpu", 2, "elapsed")]),
        ("Mixed completion", [("background", "mixed", 0, "elapsed"),
                              ("interactive", "mixed", 1, "elapsed")]),
        ("Latency target", [("response", "latency", 2, "response"),
                            ("completion", "latency", 2, "elapsed")]),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    width = 0.36
    for axis, (title, items) in zip(axes, panels):
        positions = list(range(len(items)))
        for offset, policy, color in ((-width/2, "baseline", "#777777"),
                                      (width/2, "stride", "#F58518")):
            values = [stats(records, policy, workload, role, metric)["mean"]
                      for _, workload, role, metric in items]
            axis.bar([position + offset for position in positions], values, width,
                     label=policy, color=color)
        axis.set_xticks(positions, [item[0] for item in items], rotation=15)
        axis.set_title(title)
        axis.set_ylabel("Mean ticks (lower is better)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Track G — round-robin baseline vs weighted stride (1 QEMU CPU)")
    figure.tight_layout()
    plot = run / "scheduler_comparison.png"
    figure.savefig(plot, dpi=140)
    plt.close(figure)
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["artifacts"]["analysis.json"] = {"sha256": sha256(analysis_path)}
    metadata["artifacts"]["scheduler_comparison.png"] = {"sha256": sha256(plot)}
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
