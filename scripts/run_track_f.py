"""Run Track F compiler optimization and pass-order experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_lab.benchmarking import (
    BenchmarkConfig, RunRecorder, canonical_json_bytes, numeric_summary,
    record_invalid_attempt,
)
from optimization_lab.toy_compiler import (
    PASSES, bytecode_size, compile_source, interpret, run_bytecode, select_pass_order_dp,
    serialize_ir,
)

TUNING_SOURCES = (
    "let a=2; let b=a+3; let c=b*4; print c;",
    "let x=3; let y=x+2; let p=y*y; let q=y*y; print p+q;",
    "let a=1; let dead=8*9; let b=a+2; print b;",
)

KNOWN_EVALUATION = {
    "precedence": "let x=2+3*4; print x; print -(x-10);",
    "reassignment": "let x=2; let y=x+3; print y; x=9; let z=x+3; print z;",
    "cse_invalidation": "let a=2; let b=3; let x=a+b; a=8; let y=a+b; print x; print y;",
    "dead_side_effect": "let dead=4*5; let x=2+3; print x; dead=9; print x;",
}


def generated_source(seed: int, rounds: int) -> str:
    rng = random.Random(seed)
    statements = [f"let base={rng.randint(1, 7)};", f"let scale={rng.randint(2, 5)};"]
    for index in range(rounds):
        statements.extend([
            f"let left{index}=base+scale;",
            f"let right{index}=base+scale;",
            f"let dead{index}={rng.randint(2, 9)}*{rng.randint(2, 9)};",
            f"print left{index}+right{index};",
        ])
        if index % 3 == 2:
            statements.append("base=base+1;")
    return " ".join(statements)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
               "--results-root", str(args.results_root.resolve())]
    try:
        raw_bytes = config_path.read_bytes()
        config = BenchmarkConfig.from_mapping(json.loads(raw_bytes))
        settings = config.solver_settings
        if settings.get("pipelines") != ["unoptimized", "fixed_order", "dp_order"]:
            raise ValueError("pipelines must be unoptimized, fixed_order, dp_order")
        pass_costs = {str(name): int(cost) for name, cost in settings["pass_costs"].items()}
        budget = int(settings["pass_budget"])
        rounds = int(settings["generated_assignments"])
        if rounds != config.input_size or rounds <= 0:
            raise ValueError("generated_assignments must equal positive input_size")
    except Exception as exc:
        raw_bytes = config_path.read_bytes() if config_path.is_file() else b""
        directory = record_invalid_attempt(args.results_root.resolve(), config_path,
                                           raw_bytes, exc, command)
        print(f"INVALID: {exc}\nEvidence: {directory}", file=sys.stderr)
        return 2

    recorder = RunRecorder(
        args.results_root.resolve(), config_path, config,
        source_files=[Path(__file__).resolve(), ROOT / "src/optimization_lab/toy_compiler.py",
                      ROOT / "src/optimization_lab/benchmarking.py"], command=command,
    )
    total_started = time.perf_counter()
    try:
        selection_started = time.perf_counter()
        selection = select_pass_order_dp(TUNING_SOURCES, pass_costs, budget)
        selection_seconds = time.perf_counter() - selection_started
        fixed_order = list(PASSES)
        pipelines = {
            "unoptimized": [], "fixed_order": fixed_order,
            "dp_order": selection["sequence"],
        }
        programs = [
            {"program_id": name, "kind": "known", "seed": None, "source": source}
            for name, source in KNOWN_EVALUATION.items()
        ]
        programs.extend({
            "program_id": f"generated_{seed}", "kind": "generated", "seed": seed,
            "source": generated_source(seed, rounds),
        } for seed in config.seeds)
        tuning_hashes = {hashlib.sha256(source.encode()).hexdigest() for source in TUNING_SOURCES}
        evaluation_hashes = {hashlib.sha256(item["source"].encode()).hexdigest() for item in programs}
        if tuning_hashes & evaluation_hashes:
            raise AssertionError("tuning and evaluation programs must be disjoint")

        expected_outputs = {}
        for item in programs:
            compiled = compile_source(item["source"])
            expected_outputs[item["program_id"]] = interpret(compiled["ast"])
        warmup_started = time.perf_counter()
        for _ in range(config.warmup_repetitions):
            for item in programs:
                for sequence in pipelines.values():
                    compiled = compile_source(item["source"], sequence)
                    run_bytecode(compiled["bytecode"])
        warmup_seconds = time.perf_counter() - warmup_started

        examples = []
        deadline = total_started + config.timeout_seconds
        for item in programs:
            for repetition in range(config.repetitions):
                for pipeline, sequence in pipelines.items():
                    if time.perf_counter() > deadline:
                        raise TimeoutError(f"Track F exceeded {config.timeout_seconds} seconds")
                    compile_started = time.perf_counter()
                    compiled = compile_source(item["source"], sequence)
                    compile_seconds = time.perf_counter() - compile_started
                    execute_started = time.perf_counter()
                    output, dynamic_count = run_bytecode(compiled["bytecode"])
                    execution_seconds = time.perf_counter() - execute_started
                    expected = expected_outputs[item["program_id"]]
                    if output != expected:
                        raise AssertionError(
                            f"{pipeline} changed {item['program_id']}: {output} != {expected}"
                        )
                    recorder.add_record({
                        "program_id": item["program_id"], "program_kind": item["kind"],
                        "seed": item["seed"], "repetition": repetition,
                        "pipeline": pipeline, "pass_sequence": ";".join(sequence),
                        "pass_budget_cost": sum(pass_costs[name] for name in sequence),
                        "static_instruction_count": len(compiled["bytecode"].instructions),
                        "dynamic_instruction_count": dynamic_count,
                        "bytecode_size_bytes": bytecode_size(compiled["bytecode"]),
                        "compile_seconds": compile_seconds,
                        "execution_seconds": execution_seconds,
                        "output": list(output), "reference_output": list(expected),
                        "equivalent": True,
                    })
            examples.append({
                **item, "expected_output": list(expected_outputs[item["program_id"]]),
                "pipelines": {
                    name: {
                        "sequence": sequence,
                        "ir": json.loads(serialize_ir(compile_source(item["source"], sequence)["ir"])),
                        "bytecode": compile_source(item["source"], sequence)["bytecode"].instructions,
                    } for name, sequence in pipelines.items()
                },
            })

        programs_path = recorder.directory / "evaluation_programs.json"
        selection_path = recorder.directory / "pass_selection.json"
        examples_path = recorder.directory / "compiler_examples.json"
        write_json(programs_path, programs)
        write_json(selection_path, {
            **selection, "selection_seconds": selection_seconds,
            "pass_costs": pass_costs, "fixed_order": fixed_order,
            "tuning_sources": list(TUNING_SOURCES),
            "tuning_source_hashes": sorted(tuning_hashes),
            "evaluation_source_hashes": sorted(evaluation_hashes),
        })
        write_json(examples_path, examples)

        static_summary = numeric_summary(recorder.records, "pipeline", "static_instruction_count")
        dynamic_summary = numeric_summary(recorder.records, "pipeline", "dynamic_instruction_count")
        size_summary = numeric_summary(recorder.records, "pipeline", "bytecode_size_bytes")
        compile_summary = numeric_summary(recorder.records, "pipeline", "compile_seconds")
        execution_summary = numeric_summary(recorder.records, "pipeline", "execution_seconds")
        baseline_static = static_summary["unoptimized"]["mean"]
        baseline_size = size_summary["unoptimized"]["mean"]
        reductions = {
            pipeline: {
                "static_instruction_percent": 100.0 * (baseline_static - values["mean"]) /
                                              baseline_static,
                "bytecode_size_percent": 100.0 * (baseline_size - size_summary[pipeline]["mean"]) /
                                         baseline_size,
            } for pipeline, values in static_summary.items()
        }
        plot = recorder.directory / "compiler_comparison.png"
        labels = list(pipelines)
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
        axes[0].bar(labels, [static_summary[name]["mean"] for name in labels],
                    color=["#777777", "#4C78A8", "#F58518"])
        axes[0].set(ylabel="Mean static bytecode instructions", title="Static instruction count")
        axes[1].bar(labels, [size_summary[name]["mean"] for name in labels],
                    color=["#777777", "#4C78A8", "#F58518"])
        axes[1].set(ylabel="Mean encoded size (bytes)", title="Executable bytecode size")
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
            axis.tick_params(axis="x", rotation=15)
        fig.suptitle("Track F compiler — held-out evaluation programs")
        fig.tight_layout()
        fig.savefig(plot, dpi=140)
        plt.close(fig)

        recorder.metadata["language_semantics"] = {
            "integers": "unbounded signed Python integers; no overflow",
            "evaluation": "left-to-right straight-line statements",
            "side_effects": "print is the only side effect and is never removed",
            "undefined_variables": "compile succeeds but reference and VM execution raise NameError",
        }
        recorder.metadata["timing_policy"] = {
            "clock": "time.perf_counter", "warmups_excluded": True,
            "warmup_repetitions": config.warmup_repetitions,
            "measured_repetitions": config.repetitions,
            "compile_includes": "lex, parse, IR lowering, passes and bytecode generation",
            "execution_is_separate": True,
            "pass_selection_excluded_from_compile_timing": True,
            "pass_selection_seconds": selection_seconds,
        }
        write_json(recorder.directory / "metadata.json", recorder.metadata)
        summary = {
            "status": "completed", "scope": "Track F toy compiler optimization",
            "program_count": len(programs), "known_programs": len(KNOWN_EVALUATION),
            "generated_programs": len(config.seeds), "record_count": len(recorder.records),
            "pipelines": pipelines, "pass_selection": selection,
            "static_instruction_count": static_summary,
            "dynamic_instruction_count": dynamic_summary,
            "bytecode_size_bytes": size_summary, "compile_seconds": compile_summary,
            "execution_seconds": execution_summary, "reductions_vs_unoptimized": reductions,
            "equivalence_failures": sum(not record["equivalent"] for record in recorder.records),
        }
        recorder.complete(summary, {
            "comparison_plot": plot, "evaluation_programs": programs_path,
            "pass_selection": selection_path, "compiler_examples": examples_path,
        })
        print(recorder.directory)
        return 0
    except Exception as exc:
        recorder.fail(exc)
        print(f"FAILED: {exc}\nEvidence: {recorder.directory}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
