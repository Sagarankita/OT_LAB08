"""A small, auditable compiler and optimization-pass laboratory for Track F."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Sequence


TOKEN_RE = re.compile(r"\s*(?:(\d+)|([A-Za-z_]\w*)|(.))")


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


class Expr: ...


@dataclass(frozen=True)
class IntExpr(Expr):
    value: int


@dataclass(frozen=True)
class VarExpr(Expr):
    name: str


@dataclass(frozen=True)
class BinExpr(Expr):
    op: str
    left: Expr
    right: Expr


class Statement: ...


@dataclass(frozen=True)
class Assign(Statement):
    name: str
    expression: Expr


@dataclass(frozen=True)
class Print(Statement):
    expression: Expr


def lex(source: str) -> list[Token]:
    tokens: list[Token] = []
    position = 0
    while position < len(source):
        match = TOKEN_RE.match(source, position)
        if not match:
            raise SyntaxError(f"invalid token at offset {position}")
        integer, identifier, symbol = match.groups()
        position = match.end()
        if integer is not None:
            tokens.append(Token("INT", integer))
        elif identifier is not None:
            tokens.append(Token(identifier if identifier in {"let", "print"} else "IDENT",
                                identifier))
        elif symbol in "+-*()=;":
            tokens.append(Token(symbol, symbol))
        else:
            raise SyntaxError(f"unexpected character {symbol!r}")
    tokens.append(Token("EOF", ""))
    return tokens


class Parser:
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = tokens
        self.position = 0

    def peek(self) -> Token:
        return self.tokens[self.position]

    def take(self, kind: str) -> Token:
        token = self.peek()
        if token.kind != kind:
            raise SyntaxError(f"expected {kind}, got {token.kind}")
        self.position += 1
        return token

    def parse(self) -> tuple[Statement, ...]:
        statements = []
        while self.peek().kind != "EOF":
            if self.peek().kind == "let":
                self.take("let")
                name = self.take("IDENT").value
                self.take("=")
                statements.append(Assign(name, self.expression()))
            elif self.peek().kind == "print":
                self.take("print")
                statements.append(Print(self.expression()))
            else:
                name = self.take("IDENT").value
                self.take("=")
                statements.append(Assign(name, self.expression()))
            self.take(";")
        return tuple(statements)

    def expression(self) -> Expr:
        value = self.term()
        while self.peek().kind in {"+", "-"}:
            op = self.take(self.peek().kind).kind
            value = BinExpr(op, value, self.term())
        return value

    def term(self) -> Expr:
        value = self.factor()
        while self.peek().kind == "*":
            self.take("*")
            value = BinExpr("*", value, self.factor())
        return value

    def factor(self) -> Expr:
        token = self.peek()
        if token.kind == "INT":
            return IntExpr(int(self.take("INT").value))
        if token.kind == "IDENT":
            return VarExpr(self.take("IDENT").value)
        if token.kind == "-":
            self.take("-")
            return BinExpr("-", IntExpr(0), self.factor())
        if token.kind == "(":
            self.take("(")
            value = self.expression()
            self.take(")")
            return value
        raise SyntaxError(f"expected expression, got {token.kind}")


def parse(source: str) -> tuple[Statement, ...]:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string")
    return Parser(lex(source)).parse()


def _eval_expr(expression: Expr, environment: dict[str, int]) -> int:
    if isinstance(expression, IntExpr):
        return expression.value
    if isinstance(expression, VarExpr):
        if expression.name not in environment:
            raise NameError(f"undefined variable {expression.name}")
        return environment[expression.name]
    assert isinstance(expression, BinExpr)
    left = _eval_expr(expression.left, environment)
    right = _eval_expr(expression.right, environment)
    return {"+": left + right, "-": left - right, "*": left * right}[expression.op]


def interpret(statements: Sequence[Statement]) -> tuple[int, ...]:
    environment: dict[str, int] = {}
    output: list[int] = []
    for statement in statements:
        if isinstance(statement, Assign):
            environment[statement.name] = _eval_expr(statement.expression, environment)
        else:
            assert isinstance(statement, Print)
            output.append(_eval_expr(statement.expression, environment))
    return tuple(output)


@dataclass(frozen=True)
class Operand:
    kind: str  # imm or var
    value: int | str


@dataclass(frozen=True)
class IRInstruction:
    op: str
    target: str | None
    args: tuple[Operand, ...]


def imm(value: int) -> Operand:
    return Operand("imm", int(value))


def var(name: str) -> Operand:
    return Operand("var", name)


def instruction_sources(instruction: IRInstruction) -> set[str]:
    return {str(arg.value) for arg in instruction.args if arg.kind == "var"}


def lower(statements: Sequence[Statement]) -> tuple[IRInstruction, ...]:
    instructions: list[IRInstruction] = []
    temporary = 0

    def expression(value: Expr, target: str | None = None) -> Operand:
        nonlocal temporary
        if isinstance(value, IntExpr):
            if target is None:
                return imm(value.value)
            instructions.append(IRInstruction("const", target, (imm(value.value),)))
            return var(target)
        if isinstance(value, VarExpr):
            if target is None:
                return var(value.name)
            instructions.append(IRInstruction("copy", target, (var(value.name),)))
            return var(target)
        assert isinstance(value, BinExpr)
        left = expression(value.left)
        right = expression(value.right)
        destination = target
        if destination is None:
            destination = f"$t{temporary}"
            temporary += 1
        instructions.append(IRInstruction({"+": "add", "-": "sub", "*": "mul"}[value.op],
                                          destination, (left, right)))
        return var(destination)

    for statement in statements:
        if isinstance(statement, Assign):
            expression(statement.expression, statement.name)
        else:
            assert isinstance(statement, Print)
            instructions.append(IRInstruction("print", None, (expression(statement.expression),)))
    return tuple(instructions)


def constant_folding(instructions: Sequence[IRInstruction]) -> tuple[IRInstruction, ...]:
    output = []
    for instruction in instructions:
        if instruction.op in {"add", "sub", "mul"} and all(
                arg.kind == "imm" for arg in instruction.args):
            left, right = (int(arg.value) for arg in instruction.args)
            value = {"add": left + right, "sub": left - right, "mul": left * right}[instruction.op]
            output.append(IRInstruction("const", instruction.target, (imm(value),)))
        else:
            output.append(instruction)
    return tuple(output)


def constant_propagation(instructions: Sequence[IRInstruction]) -> tuple[IRInstruction, ...]:
    constants: dict[str, int] = {}
    output = []
    for instruction in instructions:
        args = tuple(imm(constants[str(arg.value)])
                     if arg.kind == "var" and str(arg.value) in constants else arg
                     for arg in instruction.args)
        rewritten = IRInstruction(instruction.op, instruction.target, args)
        if instruction.target is not None:
            constants.pop(instruction.target, None)
            if rewritten.op == "const":
                constants[instruction.target] = int(args[0].value)
            elif rewritten.op == "copy" and args[0].kind == "imm":
                rewritten = IRInstruction("const", instruction.target, args)
                constants[instruction.target] = int(args[0].value)
        output.append(rewritten)
    return tuple(output)


def dead_code_elimination(instructions: Sequence[IRInstruction]) -> tuple[IRInstruction, ...]:
    live: set[str] = set()
    kept: list[IRInstruction] = []
    for instruction in reversed(instructions):
        if instruction.op == "print":
            kept.append(instruction)
            live.update(instruction_sources(instruction))
        elif instruction.target in live:
            kept.append(instruction)
            live.remove(str(instruction.target))
            live.update(instruction_sources(instruction))
    return tuple(reversed(kept))


def common_subexpression_elimination(
        instructions: Sequence[IRInstruction]) -> tuple[IRInstruction, ...]:
    expressions: dict[tuple[object, ...], str] = {}
    output = []
    for instruction in instructions:
        target = instruction.target
        if target is not None:
            expressions = {
                key: result for key, result in expressions.items()
                if result != target and not any(
                    isinstance(part, tuple) and part == ("var", target) for part in key
                )
            }
        if instruction.op in {"add", "sub", "mul"}:
            encoded = [(arg.kind, arg.value) for arg in instruction.args]
            if instruction.op in {"add", "mul"}:
                encoded.sort(key=repr)
            key = (instruction.op, *encoded)
            if key in expressions:
                instruction = IRInstruction("copy", target, (var(expressions[key]),))
            else:
                expressions[key] = str(target)
        output.append(instruction)
    return tuple(output)


PASSES: dict[str, Callable[[Sequence[IRInstruction]], tuple[IRInstruction, ...]]] = {
    "constant_folding": constant_folding,
    "constant_propagation": constant_propagation,
    "dead_code_elimination": dead_code_elimination,
    "common_subexpression_elimination": common_subexpression_elimination,
}


def apply_passes(instructions: Sequence[IRInstruction], sequence: Sequence[str]) -> tuple[IRInstruction, ...]:
    result = tuple(instructions)
    for name in sequence:
        if name not in PASSES:
            raise ValueError(f"unknown optimization pass {name}")
        result = PASSES[name](result)
    return result


def serialize_ir(instructions: Sequence[IRInstruction]) -> str:
    return json.dumps([
        [item.op, item.target, [[arg.kind, arg.value] for arg in item.args]]
        for item in instructions
    ], separators=(",", ":"))


def select_pass_order_dp(tuning_sources: Sequence[str], pass_costs: dict[str, int],
                         budget: int) -> dict[str, object]:
    """Dynamic programming over IR states finds the best full pass permutation."""
    names = tuple(PASSES)
    if set(pass_costs) != set(names) or any(cost <= 0 for cost in pass_costs.values()):
        raise ValueError("pass costs must contain one positive cost for every pass")
    required_cost = sum(pass_costs.values())
    if budget < required_cost:
        raise ValueError("budget must permit each required pass exactly once")
    initial = tuple(lower(parse(source)) for source in tuning_sources)
    initial_count = sum(len(program) for program in initial)
    initial_size = sum(bytecode_size(generate_bytecode(program)) for program in initial)

    @lru_cache(maxsize=None)
    def visit(used: int, states: tuple[str, ...], spent: int):
        decoded = tuple(_deserialize_ir(state) for state in states)
        if used == (1 << len(names)) - 1:
            return sum(bytecode_size(generate_bytecode(program)) for program in decoded), ()
        best: tuple[int, tuple[str, ...]] | None = None
        for index, name in enumerate(names):
            if used & (1 << index) or spent + pass_costs[name] > budget:
                continue
            transformed = tuple(PASSES[name](program) for program in decoded)
            candidate_count, suffix = visit(
                used | (1 << index), tuple(serialize_ir(program) for program in transformed),
                spent + pass_costs[name],
            )
            candidate = (candidate_count, (name, *suffix))
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise ValueError("no full pass sequence fits the budget")
        return best

    final_size, sequence = visit(0, tuple(serialize_ir(program) for program in initial), 0)
    final_programs = tuple(apply_passes(program, sequence) for program in initial)
    final_count = sum(len(program) for program in final_programs)
    return {
        "sequence": list(sequence), "budget": budget, "cost": required_cost,
        "tuning_program_count": len(tuning_sources),
        "initial_instruction_count": initial_count,
        "final_instruction_count": final_count,
        "instruction_reduction": initial_count - final_count,
        "initial_bytecode_size": initial_size,
        "final_bytecode_size": final_size,
        "bytecode_reduction": initial_size - final_size,
        "objective": "minimize total tuning bytecode bytes",
        "states_evaluated": visit.cache_info().currsize,
    }


def _deserialize_ir(value: str) -> tuple[IRInstruction, ...]:
    return tuple(IRInstruction(op, target, tuple(Operand(kind, operand) for kind, operand in args))
                 for op, target, args in json.loads(value))


@dataclass(frozen=True)
class BytecodeProgram:
    instructions: tuple[tuple[object, ...], ...]
    slot_names: tuple[str, ...]


def generate_bytecode(instructions: Sequence[IRInstruction]) -> BytecodeProgram:
    names = sorted({name for instruction in instructions
                    for name in ([instruction.target] if instruction.target else []) +
                    list(instruction_sources(instruction))})
    slots = {name: index for index, name in enumerate(names)}

    def encode(operand: Operand):
        return ("i", int(operand.value)) if operand.kind == "imm" else ("s", slots[str(operand.value)])

    code = []
    for instruction in instructions:
        if instruction.op == "print":
            code.append(("PRINT", encode(instruction.args[0])))
        elif instruction.op == "const":
            code.append(("CONST", slots[str(instruction.target)], int(instruction.args[0].value)))
        elif instruction.op == "copy":
            code.append(("COPY", slots[str(instruction.target)], encode(instruction.args[0])))
        else:
            code.append((instruction.op.upper(), slots[str(instruction.target)],
                         encode(instruction.args[0]), encode(instruction.args[1])))
    return BytecodeProgram(tuple(code), tuple(names))


def run_bytecode(program: BytecodeProgram) -> tuple[tuple[int, ...], int]:
    slots = [0] * len(program.slot_names)
    initialized = [False] * len(program.slot_names)
    output = []

    def read(operand) -> int:
        kind, value = operand
        if kind == "i":
            return int(value)
        if not initialized[int(value)]:
            raise NameError(f"read from uninitialized slot {value}")
        return slots[int(value)]

    executed = 0
    for instruction in program.instructions:
        executed += 1
        op = instruction[0]
        if op == "PRINT":
            output.append(read(instruction[1]))
        elif op == "CONST":
            slots[int(instruction[1])] = int(instruction[2])
            initialized[int(instruction[1])] = True
        elif op == "COPY":
            slots[int(instruction[1])] = read(instruction[2])
            initialized[int(instruction[1])] = True
        else:
            left, right = read(instruction[2]), read(instruction[3])
            slots[int(instruction[1])] = {"ADD": left + right, "SUB": left - right,
                                          "MUL": left * right}[op]
            initialized[int(instruction[1])] = True
    return tuple(output), executed


def bytecode_size(program: BytecodeProgram) -> int:
    return len(json.dumps({"slots": program.slot_names, "code": program.instructions},
                          separators=(",", ":")).encode("utf-8"))


def compile_source(source: str, passes: Sequence[str] = ()) -> dict[str, object]:
    ast = parse(source)
    unoptimized = lower(ast)
    optimized = apply_passes(unoptimized, passes)
    bytecode = generate_bytecode(optimized)
    return {"ast": ast, "unoptimized_ir": unoptimized, "ir": optimized, "bytecode": bytecode}
