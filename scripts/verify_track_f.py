"""Independently verify Track F pass selection, bytecode and measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization_lab.benchmarking import load_completed_run, sha256_file
from optimization_lab.toy_compiler import (
    PASSES, bytecode_size, compile_source, interpret, select_pass_order_dp,
)


def independent_vm(program):
    slots = [None] * len(program.slot_names)
    output = []

    def read(operand):
        kind, value = operand
        if kind == "i":
            return int(value)
        if slots[int(value)] is None:
            raise NameError("uninitialized slot")
        return slots[int(value)]

    executed = 0
    for instruction in program.instructions:
        executed += 1
        op = instruction[0]
        if op == "PRINT":
            output.append(read(instruction[1]))
        elif op == "CONST":
            slots[int(instruction[1])] = int(instruction[2])
        elif op == "COPY":
            slots[int(instruction[1])] = read(instruction[2])
        else:
            left, right = read(instruction[2]), read(instruction[3])
            slots[int(instruction[1])] = {"ADD": left + right, "SUB": left - right,
                                          "MUL": left * right}[op]
    return tuple(output), executed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    loaded = load_completed_run(run)
    config, metadata, records, summary = (
        loaded["config"], loaded["metadata"], loaded["records"], loaded["summary"]
    )
    programs = json.loads((run / "evaluation_programs.json").read_text())
    selection = json.loads((run / "pass_selection.json").read_text())
    recomputed_selection = select_pass_order_dp(
        selection["tuning_sources"], selection["pass_costs"], selection["budget"]
    )
    pipelines = summary["pipelines"]
    expected_records = len(programs) * len(pipelines) * config["repetitions"]
    paired = all(
        {record["pipeline"] for record in records
         if record["program_id"] == item["program_id"] and
         record["repetition"] == repetition} == set(pipelines)
        for item in programs for repetition in range(config["repetitions"])
    )
    record_lookup = {(record["program_id"], record["pipeline"], record["repetition"]): record
                     for record in records}
    semantics_ok = True
    counts_ok = True
    for item in programs:
        reference = interpret(compile_source(item["source"])["ast"])
        for pipeline, sequence in pipelines.items():
            compiled = compile_source(item["source"], sequence)
            output, dynamic_count = independent_vm(compiled["bytecode"])
            semantics_ok &= output == reference
            for repetition in range(config["repetitions"]):
                record = record_lookup[(item["program_id"], pipeline, repetition)]
                semantics_ok &= tuple(record["output"]) == reference == tuple(record["reference_output"])
                counts_ok &= (
                    record["static_instruction_count"] == len(compiled["bytecode"].instructions)
                    and record["dynamic_instruction_count"] == dynamic_count
                    and record["bytecode_size_bytes"] == bytecode_size(compiled["bytecode"])
                    and record["pass_sequence"] == ";".join(sequence)
                )
    tuning_hashes = {hashlib.sha256(source.encode()).hexdigest()
                     for source in selection["tuning_sources"]}
    evaluation_hashes = {hashlib.sha256(item["source"].encode()).hexdigest() for item in programs}
    fixed_tuning_size = sum(bytecode_size(compile_source(source, selection["fixed_order"])["bytecode"])
                            for source in selection["tuning_sources"])
    status = json.loads((run / "status.json").read_text())
    hashes_match = all(
        sha256_file(run / artifact["path"]) == artifact["sha256"]
        for artifact in status["artifacts"].values()
    )
    checks = {
        "final_configuration": config["run_kind"] == "final",
        "five_generated_seeds": len(config["seeds"]) == 5,
        "known_and_generated_evaluation_programs":
            {item["kind"] for item in programs} == {"known", "generated"},
        "record_count": len(records) == expected_records == summary["record_count"],
        "all_pipeline_pairs_present": paired,
        "all_outputs_match_ast_reference_and_independent_vm": semantics_ok,
        "instruction_counts_sizes_and_sequences_recomputed": counts_ok,
        "all_four_passes_in_fixed_and_dp_sequences":
            set(pipelines["fixed_order"]) == set(PASSES) == set(pipelines["dp_order"]),
        "pass_budget_equal":
            sum(selection["pass_costs"][name] for name in pipelines["fixed_order"]) ==
            sum(selection["pass_costs"][name] for name in pipelines["dp_order"]) ==
            selection["budget"],
        "dp_selection_reproduced":
            selection["sequence"] == recomputed_selection["sequence"] and
            selection["final_bytecode_size"] == recomputed_selection["final_bytecode_size"],
        "dp_tuning_objective_no_worse_than_fixed":
            selection["final_bytecode_size"] <= fixed_tuning_size,
        "tuning_and_evaluation_sets_disjoint": not (tuning_hashes & evaluation_hashes),
        "reassignment_and_side_effect_cases_present":
            {"reassignment", "cse_invalidation", "dead_side_effect"}.issubset(
                {item["program_id"] for item in programs}
            ),
        "timing_policy_recorded": metadata["timing_policy"]["warmups_excluded"] is True,
        "integer_and_side_effect_semantics_recorded":
            metadata["language_semantics"]["side_effects"].startswith("print"),
        "all_declared_artifact_hashes_match": hashes_match,
        "git_provenance_pending_user":
            metadata["provenance"]["git_revision"] == "pending user Git work",
    }
    report = {
        "checks": checks, "passed": all(checks.values()), "raw_records": len(records),
        "evaluation_programs": len(programs), "selected_sequence": selection["sequence"],
        "fixed_tuning_bytes": fixed_tuning_size,
        "dp_tuning_bytes": selection["final_bytecode_size"],
    }
    (run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
