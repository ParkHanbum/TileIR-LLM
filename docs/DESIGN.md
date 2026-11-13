# Design: TileIR runtime research and its GEMM library

## Purpose and boundary

The parent project aims to build a TileIR runtime with end-to-end performance
comparable to vLLM while preserving observable model behavior. TileIRBLAS is its
GEMM subproject, targeting the functionality and performance of the public cuBLAS
operations actually used by the model. The runtime goal also includes non-GEMM
operators, state ownership, scheduling integration, and host orchestration; it
cannot be established by GEMM throughput alone.

The current implementation investigates that goal through a fixed workload:
Qwen3.5-4B in vLLM, with batch size 32, prompt length 53, and 16 generated tokens
per request. The qualified configuration uses BF16, eager execution, compilation
mode 0, CUDA graphs disabled, and five public logprob entries. The pinned precision
policy enables both BF16 reduced-precision reduction flags (T/T).

The unit of compatibility is an observed operation contract, not a vendor binary
interface. Public vLLM and PyTorch behavior, public cuBLAS reference calls, cuTile
interfaces, and published techniques such as CUTLASS tiling inform the work. This
is not vendor-binary reverse engineering. Reference implementations are used to
check behavior; they are not silently invoked to make an unsupported candidate
operation pass.

The result is a research library with workload-specific Python adapters. It is
not a drop-in cuBLAS shared library, a general vLLM backend, or proof of support
for arbitrary shapes, models, devices, training, or CUDA graphs. Performance
parity with the native implementation has not been established.

## Current engineering method

Work starts at observed execution regions and module-level public APIs. We
compare their contracts and behavior with our own implementation logic, including
numerical operations, layout and alias rules, state updates, and host-side
lifetime obligations. This is implementation and runtime integration work, not
an investigation of proprietary vendor internals.

Changes are evaluated individually. First establish affected-operation and model
correctness, then re-establish the same-request compute census and scoped
no-cuBLAS evidence. Only after those gates pass is a separate warm regression
meaningful. This order prevents a faster but semantically different computation,
or diagnostic overhead, from being mistaken for runtime progress.

TileIR/LLVM IR compiler-pass research is not part of the current scope. It remains
a possible later research direction with its own design and validation needs.
The present project does not claim compiler-pass changes or compiler-level
performance improvements.

## Source identities

Two identities have different roles:

- `v7-own-kernel-fixed-workload-20261113`, commit
  `0b9611661ef216c35fb9f4c5da92662e5930dcbb`, preserves the numerical baseline.
- `f9c69c83c2edc99011e036a885aedf38049d80c2` is the runnable regression source
  revision described here. Later regression support does not move the baseline
  tag or turn it into a self-contained runnable checkout.

An execution claim needs receipts for the exact full revision, its admitted
sources and dependencies, and the actual process exits. A source commit is not
itself an execution receipt. Exploratory G/H/O and other performance branches
are not part of the original-v7 baseline or its coverage claims.

## Layers

```text
Pinned model, inputs, runtime, and source revision
                         |
                         v
vLLM eager engine: scheduling, prefill, decode, public outputs
                         |
                         v
Scoped runtime adapters
  install / binding checks / request counters / failure latch
  state and storage ownership / restore / synchronized retirement
                         |
                         v
Functional operation APIs
  dtype / shape / stride / offset / range / alias validation
  allocation and output semantics / current-stream enqueue
                         |
                         v
Own TileIR kernels
  GEMM / recurrence / convolution / attention / pointwise / guards
                         |
                         v
CUDA execution and ordinary runtime transport
```

Kernel code implements numerical work. API code specifies which tensors and
metadata are supported and what outputs or state updates mean. Runtime adapters
connect those APIs to the selected vLLM call sites. These responsibilities must
not be collapsed: a correct kernel can still be incorrectly bound, passed the
wrong view, or allowed to outlive its storage.

For example, `tileir_vllm_gemma_norm_v3/api.py` explicitly exposes fresh functional
outputs rather than the native in-place fused-normalization ABI. It validates
inference tensors, supported dimensions, strides, epsilon, and device before
launching its kernels. An adapter must account for that distinction instead of
assuming matching operation names imply matching mutation behavior.

The whole-path coordinator is `whole_lean_extensions_runtime_v7/adapter.py`,
with the additional domains assembled in `extension.py`. It composes existing
versioned implementations rather than providing one monolithic model kernel.
Installation and cleanup are part of the contract, including partial failures.

## Model execution

```text
B32 token inputs, P53
       |
       v
Prefill: 1696 token positions
  embeddings -> repeated model layers
                 |-- projections and normalization
                 |-- attention + KV cache, or gated delta recurrence
                 |-- causal convolution and state preparation where required
                 `-- residual / activation / output transformations
       |
       v
Decode: 15 follow-up steps, 32 positions per step
  cached/stateful layer updates -> vocabulary scores
       |
       v
sampling and selected public logprobs
  first generated token comes from prefill
       |
       v
512 generated token IDs + 2560 public values checked
```

This is a functional overview, not a claim that every layer has the same
attention or recurrence structure. Prefill and decode have different layouts,
state transitions, and launch counts. The adapter census records those
boundaries separately from the GPU trace's physical kernel launches.

Fresh allocations, source read-only guarantees, selected-state writes, and
permitted aliases are operation-specific. A compact standalone state fixture
must not be described as a replay of the original full shared allocation.
Device storage and mapped host/UVA storage also need different lifetime
reasoning. Strong references, verified owner identity, synchronization, and
restoration checks cannot be replaced by a successful output comparison.

Unsupported inputs are rejected within the documented API scope. Failure
latches prevent subsequent work from proceeding as if an earlier operation had
succeeded. Cleanup must preserve foreign bindings and must not release owners
when restoration or synchronization is uncertain.

## What the evidence proves

Numerical and operational checks are deliberately separate:

| Evidence | Meaning | Not implied |
|---|---|---|
| Raw comparison | Exact bits for the stated outputs or state fixture | All inputs or every model intermediate |
| Strict/tolerance comparison | The specified numerical comparison passed | Raw equality unless checked separately |
| Finite/range/equality checks | The predicate and its error contract were checked | Safe ownership or supported layout |
| Read-only, alias, state checks | Required mutation and storage rules held in that test | Arbitrary external storage compatibility |
| Whole endpoints | Fixed token IDs and public values matched | Every intermediate tensor was compared |
| Physical census | Correlated request compute launches were attributed | Every CPU/API/transfer operation is TileIR |

Historical GEMM raw evidence covers 13 contracts and 2448 calls, not universal
cuBLAS equivalence. The dedicated fresh GEMM suite uses three actual saved cases
and fresh public references. Other module suites mix captured inputs with
explicitly bounded synthetic fixtures. `regression_v7/COVERAGE.md` identifies
which operations have dedicated cases and which still rely on whole-model,
CPU-contract, or historical evidence.

Tests include deliberately corrupted returned copies and malformed receipts.
A negative test must exercise the actual comparator or gate; showing that a
modified value differs from its source is not sufficient. NaN, finite values,
raw payloads, and numerical tolerances must retain their distinct meanings.

## Compute attribution and the no-cuBLAS claim

The original fixed-request evidence attributes 13300 compute launches to 168
full symbols across 58 kernel source files. This is a compute denominator, not
an assertion that all runtime activity has been replaced.

`regression_v7/whole.py` joins request-scoped CPU launch correlations with GPU
trace events, exact own ELF symbols, generated bytecode, and pinned source
bytes. Symbol names or a surrounding scope label alone are insufficient.
Unknown or ambiguous attribution prevents a full-coverage result. Transfers,
memset, CPU synchronization, and allocation remain separate runtime concerns;
model copy/initialization operations must not be hidden by calling them
transport.

The public logger in `whole_v7_public_blas_logger_v4/logger.py` uses official
classic cuBLAS and cuBLASLt logging interfaces. The native request is a positive
control, including the expected 2448 classic GEMM calls. The lean request must
have no observed classic cuBLAS API calls, and the whole audit also checks the
Lt evidence and logger controls. Lt coverage is experimental and must retain
its reported limitations. This is scoped to the instrumented execution,
not a universal statement about every process or configuration.

Own CUBIN hashes also link to static local-call evidence. Exact prior hashes may
reuse prior audits; new artifacts need new own-output analysis. Vendor binaries
are not inspected. Source/symbol/artifact matching does not observe the SHA of
the module actually loaded by the driver. That stronger loaded-module identity
claim remains explicitly unproven.

## Regression gate architecture

```text
Exact HEAD + clean tracked sources + immutable baseline checks
                              |
                              v
CPU suites and deliberate failures (GPU hidden)
                              |
                              v
Explicit external artifact/cache preparation
                              |
                              v
Affected standalone GPU modules
  fixture hashes -> execution -> strict module result audit
                              |
                              v
Whole endpoint + dispatch + trace/logger/source-artifact audit
                              |
                              v
Actual child and outer exits + independent recovery
                              |
                              v
Separate repeat qualification, then separate performance work
```

`regression_v7/common.py` owns source and external-dependency admission.
`cli.py` exposes explicit CPU, prepare, GPU, and analyze operations. Its ten
module groups have dedicated result audits; a generic `complete` flag is not
enough. `whole.py` reuses the original request and launcher structure, while
`recovery.py` checks the independent post-run state. GPU work is bounded and
serialized, and failed runs retain partial results. Preparing sources is not
permission to run a GPU test, and an incomplete result is never promoted to
PASS by cleanup alone.

See `baseline_v7/README.md` for provenance and excluded external evidence,
`regression_v7/README.md` for the replay interface,
`regression_v7/COVERAGE.md` and `catalog.json` for scope, and
`regression_v7/WORKFLOW.md` for change qualification. Repeat lifecycle and warm
measurement are separate gates, not consequences of a one-shot endpoint PASS.

## Reproducibility and distribution limits

The current replay is environment-bound and nonrelocatable. Saved input banks,
model/checkpoint files, installed runtimes and native libraries, and historical
evidence are external hash-bound dependencies. A source-only clone does not
contain them. Do not publish tensor payloads or private evidence as portable test data.
Machine-specific path examples do not make those dependencies available. A portable package would require
an explicit dependency and fixture design, not simply removing path checks.

Preserve per-component license and attribution notices, including those carried
with reused public sources and techniques. CUTLASS, cuTile, vLLM, and other
components have their own terms. Examples of tracked notices are
`attention_gate_runtime_integration_v1/VLLM-LICENSE`,
`runtime_metadata_five_v1/LICENSE.vllm`,
`tileir_gdn_chunk_output_v1/LICENSE-FLA-MIT.txt`, and
`public_fa2_arithmetic_v1/LICENSE`. The f9 Git tree has no root `LICENSE`
or tracked `licenses/CUTLASS-BSD-3-Clause.txt`; a local untracked notice is not
a bundled distribution license. A source-by-source notice and manifest audit
is required before publication. This archive is not declared distribution-ready.
Source provenance and public availability do
not establish a blanket redistribution license; review the actual files and
applicable terms before distribution. This document makes no blanket legal
compatibility conclusion.

## Change principles and non-goals

- Preserve proven originals and failures; version intentional changes.
- Keep numerical, API, binding, lifetime, and reporting changes distinguishable.
- Do not relax an error, range, precision, or alias check to obtain a PASS.
- Require exact-source admission and fresh affected-module evidence before whole
  qualification. Never substitute an old receipt for a new implementation.
- Keep diagnostic overhead out of performance conclusions. Profilers, loggers,
  source checks, and generated-code validation can affect host cost differently
  across implementations; diagnostic durations are not causal wall-time savings.
- Treat speed, functional coverage, raw equality, and portability as separate
  outcomes. No performance-equivalence claim follows from this design.
