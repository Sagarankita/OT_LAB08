import pytest

from optimization_lab.toy_compiler import (
    PASSES, apply_passes, bytecode_size, compile_source, interpret, parse, run_bytecode,
    select_pass_order_dp,
)


ALL = tuple(PASSES)


def outputs(source, passes=()):
    compiled = compile_source(source, passes)
    return interpret(compiled["ast"]), run_bytecode(compiled["bytecode"])[0]


def test_parser_precedence_and_unary_minus():
    assert outputs("let x = 2 + 3 * 4; print x; print -(x-10);") == ((14, -4), (14, -4))


def test_all_passes_preserve_reassignment_and_print_side_effects():
    source = "let x=2; let y=x+3; print y; x=9; let z=x+3; print z; y=99; print y;"
    expected, actual = outputs(source, ALL)
    assert expected == actual == (5, 12, 99)


def test_cse_invalidates_on_operand_reassignment():
    source = "let a=2; let b=3; let x=a+b; a=8; let y=a+b; print x; print y;"
    assert outputs(source, ("common_subexpression_elimination",)) == ((5, 11), (5, 11))


def test_dce_keeps_values_needed_by_print_and_removes_dead_assignment():
    source = "let dead=4*5; let x=2+3; print x; dead=9;"
    compiled = compile_source(source, ("dead_code_elimination",))
    assert len(compiled["ir"]) < len(compiled["unoptimized_ir"])
    assert run_bytecode(compiled["bytecode"])[0] == (5,)


def test_constant_propagation_then_folding_reduces_code_after_dce():
    source = "let a=2; let b=a+3; let c=b*4; print c;"
    passes = ("constant_propagation", "constant_folding", "dead_code_elimination")
    compiled = compile_source(source, passes)
    assert run_bytecode(compiled["bytecode"])[0] == (20,)
    assert len(compiled["ir"]) < len(compiled["unoptimized_ir"])


def test_unknown_variable_is_rejected_by_both_semantic_routes():
    compiled = compile_source("print missing;")
    with pytest.raises(NameError):
        interpret(compiled["ast"])
    with pytest.raises(NameError):
        run_bytecode(compiled["bytecode"])


def test_dp_order_uses_all_passes_within_budget_and_is_deterministic():
    tuning = [
        "let a=2; let b=a+3; let c=b*4; print c;",
        "let x=3; let y=x+2; let p=y*y; let q=y*y; print p+q;",
    ]
    first = select_pass_order_dp(tuning, {name: 1 for name in PASSES}, 4)
    second = select_pass_order_dp(tuning, {name: 1 for name in PASSES}, 4)
    assert first == second
    assert set(first["sequence"]) == set(PASSES)
    assert first["cost"] == 4


def test_bytecode_size_is_deterministic_and_nonzero():
    program = compile_source("let x=2+3; print x;", ALL)["bytecode"]
    assert bytecode_size(program) == bytecode_size(program) > 0


def test_syntax_validation():
    with pytest.raises(SyntaxError):
        parse("let x = 2 print x;")
    with pytest.raises(ValueError):
        parse("  ")
