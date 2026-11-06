# Track F — toy compiler optimization

## Compiler pipeline and language

The compiler accepts a small straight-line language with `let` declarations, reassignment, integer literals, variables, parentheses, unary minus, `+`, `-`, `*`, and `print` statements:

```text
let x = 2 + 3 * 4;
print x;
x = x - 1;
print x;
```

The pipeline is lexer → recursive-descent parser → AST → three-address IR → optimization passes → slot-based executable bytecode. A separate AST interpreter supplies the reference semantics. Integers are unbounded signed Python integers; statements execute left to right; `print` is the only side effect and may not be deleted or reordered. Reading an undefined variable raises `NameError` in both semantic routes.

The bytecode VM supports `CONST`, `COPY`, `ADD`, `SUB`, `MUL`, and `PRINT`. Encoded executable size is the length in bytes of its deterministic compact JSON representation. This is a documented toy target, not native machine code.

## Four optimization passes

1. **Constant folding** replaces arithmetic whose operands are both literals with `CONST`.
2. **Constant propagation** substitutes known constants through later uses and invalidates knowledge when a name is reassigned.
3. **Dead-code elimination** performs backward liveness and removes unused assignments while retaining `PRINT` and everything it depends on.
4. **Common-subexpression elimination** reuses repeated arithmetic but invalidates expressions when an operand or saved result is reassigned. Addition and multiplication keys are canonicalized as commutative; subtraction is not.

Tests explicitly cover reassignment, CSE invalidation, dead stores, prints, precedence, unary minus, undefined variables and deterministic bytecode size.

## Pass-order optimization formulation

Decision variable: a permutation of the four passes. Every pass has cost 1 and the compile-time budget is 4, so both compared optimized pipelines execute exactly the same four passes once. The objective is to minimize total encoded bytecode bytes on three tuning programs.

Dynamic programming memoizes states consisting of the used-pass mask, accumulated cost and the transformed tuning IRs. A transition applies one unused pass. This state is necessary because pass effects are order-dependent; remembering only the current instruction count would discard transformations that unlock later passes.

The fixed order is:

```text
folding → propagation → DCE → CSE
```

The selected order is:

```text
CSE → propagation → folding → DCE
```

The selector evaluated 26 memoized states. On the tuning set, fixed order produced 319 bytes and the selected order produced 283 bytes. The tuning programs are content-hash disjoint from all evaluation programs.

## Experimental protocol

Selected evidence: `results/track_f_compiler/20261027T140035.600751Z`.

- Evaluation: four hand-written edge-case programs and five seeded generated programs.
- The generated programs contain 16 rounds of duplicate expressions, dead assignments, outputs and periodic reassignment.
- Pipelines: unoptimized, fixed order and DP-selected order.
- Two unmeasured warm-ups and ten measured compile/execute repetitions per program/pipeline.
- Total: 270 raw records.
- Compile timing includes lexing, parsing, lowering, passes and bytecode generation. Pass-order search is separately measured and excluded.
- Execution timing contains VM execution only.
- Every optimized output is compared with the AST interpreter. The independent verifier also executes bytecode with a separately implemented VM.

## Results

Means cover the nine held-out programs and ten repetitions. The wide standard deviations primarily reflect differences in program size, not just timing noise.

| Pipeline | Mean static/dynamic instructions | Mean bytecode size | Mean compile time | Mean execution time |
|---|---:|---:|---:|---:|
| Unoptimized | 51.000 | 1643.89 B | 0.3258 ms | 0.01301 ms |
| Fixed order | 40.000 | 1167.33 B | 0.3831 ms | 0.00898 ms |
| DP order | 40.222 | 1153.11 B | 0.4684 ms | 0.00863 ms |

Because this language is straight line, static and dynamic instruction counts are equal. Relative to unoptimized bytecode:

- fixed order reduced instructions by 21.57% and size by 28.99%;
- DP order reduced instructions by 21.13% and size by 29.85%.

The held-out result matches the selected objective: DP produced the smallest bytecode, but fixed order produced 0.222 fewer mean instructions. DP compilation also cost more. VM timings suggest lower execution cost for both optimized variants, but individual runs are only microseconds and therefore timing noise is substantial; instruction count and byte size are the stronger evidence.

Pass selection itself is tuning overhead and should be amortized when compiling many programs. No claim is made that the chosen order generalizes to other languages or workloads.

## Correctness and limitations

All 17 independent checks passed. They reproduce DP selection, enforce equal pass budgets, confirm tuning/evaluation separation, regenerate bytecode sizes and instruction counts, and compare all outputs through the AST interpreter and independent VM. The project suite has 47 passing tests.

Limitations include no branches, loops, functions, memory aliasing, exceptions or fixed-width overflow. Consequently, the analyses are deliberately simpler than LLVM/GCC passes. CSE is local to one straight-line instruction sequence, and compile/execution timings do not predict native-code performance.

```sh
.venv/bin/python -m pytest
.venv/bin/python scripts/run_track_f.py --config configs/track_f_debug.json
.venv/bin/python scripts/run_track_f.py --config configs/track_f_final.json
.venv/bin/python scripts/verify_track_f.py results/track_f_compiler/20261027T140035.600751Z
```
