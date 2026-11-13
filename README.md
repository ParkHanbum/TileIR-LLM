# TileIR runtime research and TileIRBLAS

The parent project aims to build a TileIR-based inference runtime with end-to-end
performance comparable to vLLM while preserving the behavior of the real model
execution. Qwen3.5-4B is the current fixed-workload target, not a claim of general
model support or an already-achieved performance result.

TileIRBLAS is the GEMM subproject: it targets the functionality and performance of
the public cuBLAS operations actually used by that workload. The broader runtime
also needs recurrence, attention, normalization, sampling, state management, and
host orchestration. Matching GEMM alone does not establish end-to-end parity.

The current method compares execution regions and module-level public APIs with
our own implementation logic, then evaluates one change at a time: correctness,
compute census and no-cuBLAS evidence, followed by separate warm regression.
TileIR/LLVM IR compiler-pass research is outside the current scope; it may be a
later direction, not the method or result claimed by this checkpoint.

The current checkpoint is a **fixed-workload research implementation and regression
archive**, not a pip-installable package or drop-in cuBLAS ABI replacement. Its
qualified eager workload has complete own-source compute-kernel attribution, but
that is not universal API coverage, a portable release, or performance parity.

- [Purpose and detailed design](docs/DESIGN.md)
- [Original baseline and provenance](baseline_v7/README.md)
- [Existing regression instructions](regression_v7/README.md)
- [Dedicated module versus whole-model coverage](regression_v7/COVERAGE.md)
- [Change and performance gates](regression_v7/WORKFLOW.md)

The existing baseline and regression documents remain unchanged. This guide adds
a top-level entry point and distinguishes a source-only smoke test from the
machine-bound full replay.

## 1. What is established

The recorded tested regression revision is:

```text
f9c69c83c2edc99011e036a885aedf38049d80c2
```

The immutable provenance tag is different:

```text
v7-own-kernel-fixed-workload-20261113
  -> 0b9611661ef216c35fb9f4c5da92662e5930dcbb
```

The tag preserves original v7 source. Later unchanged facade/import dependencies
and executable tests were added to the regression revision. Do not move the tag
or assume its original checkout is the runnable regression revision.

Recorded local results for `f9c69c8`:

| Check | Recorded result | Scope |
|---|---|---|
| CPU catalog | 56 suites, 411 tests: 410 passed, 1 explicit historical skip | Exact revision; not every possible input |
| Standalone GPU | All 10 module groups passed | Captured and bounded synthetic cases, not every kernel independently |
| Whole endpoints | 512 generated token IDs and 2560 public logprob values matched exactly | One fixed request batch |
| Compute attribution | 13300 / 13300 request-scoped compute launches, 168 full symbols, 58 kernel source files | Not CPU work, transfers, allocation, or initialization outside the request |
| Public classic cuBLAS logger | Native: 2448 `cublasGemmEx`; TileIR request: no observed classic API calls | Positive control and request windows, not all process activity |
| Own compiled artifacts | Same 172 CUBIN hashes as the prior 65-local-call-site audit | Driver-loaded module SHA was not observed |
| Process completion | Actual child/outer exit 0 and independent recovery passed | Saved local receipts; cleanup alone is not correctness |

The cuBLASLt logger has separate create/destroy controls and experimental logging
limitations; these are not a positive control for every possible Lt compute API.
The whole audit combines the reported logger evidence with the kernel census.
No claim of a vendor library call inside our kernels is inferred merely from
kernel names. See [DESIGN](docs/DESIGN.md) for the evidence chain and limitations.

These results are historical local receipts, not tests executed by reading this
README. Large receipts, traces, tensor banks, model weights, and compiled caches
are excluded from Git. Performance equivalence has **not** been established at
this revision. Do not use diagnostic or profiled request durations as warm
inference benchmarks. G/H/O and other performance experiments are not this
baseline and are not included in these figures.

## 2. Workload and environment

The qualified workload is deliberately narrow:

```text
Model:          Qwen/Qwen3.5-4B, text-only
Model revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
Input:          one frozen 53-token input sequence repeated across 32 requests
Generation:     16 tokens per request; greedy, token-only, top-5 public logprobs
Forward calls:  one prefill + 15 decode steps per request batch
Output:         32 * 16 = 512 tokens, including the first token from prefill
Dtype:          BF16 inputs/weights, operation-specific accumulators and states
Execution:      one GPU, eager, compilation mode 0, CUDA graphs NONE
```

This is not a diverse benchmark dataset or an all-prompt accuracy evaluation.
Use the supplied hash-bound input IDs; re-tokenizing a similar prompt is not the
same fixture. Multimodal inputs, speculative decoding, distributed execution,
training, other batch/sequence lengths, and graph-enabled execution are outside
this qualification.

Recorded engine settings include `max_model_len=128`, `max_num_seqs=32`,
`max_num_batched_tokens=2048`, `gpu_memory_utilization=0.85`, prefix caching off,
chunked prefill off, tensor parallel size 1, and async scheduling enabled.
The model is local and `trust_remote_code=False`. Sampling uses temperature 0,
minimum/maximum 16 tokens, `ignore_eos=False`, `detokenize=False`, `logprobs=5`,
no prompt logprobs, and seed 20261111.

Both PyTorch flags below remain enabled and are checked/restored, not silently
changed for the comparison:

```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k
```

Permission for reduced-precision reduction is not proof of the internal reduction
dtype or accumulation order chosen for every reference call.

| Component | Qualified environment |
|---|---|
| OS / GPU | Linux x86-64, NVIDIA GeForce RTX 4080, 16376 MiB reported VRAM |
| NVIDIA driver | 595.99.02 |
| Python | 3.14.4 |
| PyTorch runtime | 2.13.0+cu130 |
| vLLM | 0.32.0 |
| cuda-tile / cuTile | 1.7.0 |
| tileiras | 13.4.92 |
| Transformers | 5.19.0 |
| Triton | 3.7.1 |
| CUDA runtime package | 13.0.96 |
| cuBLAS package | 13.1.1.3 |

This is a version record, not an installation lockfile or a supported-version
matrix. Matching version strings alone is insufficient: the replay also checks
source and external-file hashes. The desktop GUI remained active; results do
not establish exclusive use of the GPU.

## 3. Build and dependency model

There is no root `pyproject.toml`, `setup.py`, requirements lockfile, CMake build,
or complete fresh-machine installer in this checkpoint. Do not interpret a
`pip install` of similarly named versions as reproduction of this environment.

Python APIs launch cuTile kernels, which are compiled by the installed toolchain
and cached at runtime. The qualified replay uses a verified cache preseed in a
new per-run directory. Compilation and cache preparation are separate from
performance claims. A supported portable package and reproducible installer are
future work, not hidden prerequisites that this guide claims to provide.

Full replay additionally requires the original, hash-matching:

- Model/tokenizer snapshot and installed runtime/native libraries.
- Frozen token IDs, captured tensor banks, and per-module fixture metadata.
- Historical prerequisite receipts and source/license provenance files.
- Cache source named by the pinned preseed receipt: 1033 files, 57498718 bytes.
- Linux `/proc`, the coordinator lock, and machine-specific admission helper.

The dependency inventory is in `baseline_v7/EXTERNAL-EVIDENCE.json`,
`baseline_v7/BASELINE.json`, the module `dependencies()` functions, and their
referenced manifests. Git contains references/hashes, not those payloads.
There is no automatic download, replacement fixture generator, or permissive
fallback when a required artifact is missing.

## 4. Source-only smoke test (no GPU, model, or private fixtures)

From a source checkout, using Python 3.14.4 as tested:

```sh
cd /path/to/TileIRBLAS
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python3 -B -m unittest regression_v7.test_performance_decision
```

This small standard-library test runs nine cases for the statistical decision function,
including boundary and invalid-input cases. It does **not** build a GPU kernel,
qualify an implementation, validate the external environment, or report model
performance. It is intentionally usable separately from the full replay gates.

## 5. Full CPU catalog on the qualified workstation

The following are actual paths used by the environment-bound replay. They are
examples for that workstation, not paths a fresh clone creates automatically.
Do not run the full replay from an arbitrary docs/performance worktree or merely
rewrite the pinned paths to make admission pass.

```sh
REPO=/home/n/work/TileIR/public-tileir-blas-20261111
PY=/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python
HEAD=f9c69c83c2edc99011e036a885aedf38049d80c2
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1

test "$(git rev-parse HEAD)" = "$HEAD" || exit 1
test "$(git rev-parse 'v7-own-kernel-fixed-workload-20261113^{commit}')" = \
  0b9611661ef216c35fb9f4c5da92662e5930dcbb || exit 1
test -z "$(git diff --name-only HEAD --)" || exit 1

CUDA_VISIBLE_DEVICES='' "$PY" -B -m regression_v7.cli --head "$HEAD" --cpu
```

The full catalog runs each test module in a separate subprocess. GPU visibility
is disabled. Some CPU tests still read environment-specific sources and metadata,
so they are not guaranteed to pass on a source-only clone. The exact one-test
historical packaging skip is explained in `regression_v7/README.md`; no cache is
deleted and no unrelated failure is converted to a skip.

For a focused CPU test, the CLI also accepts:

```sh
CUDA_VISIBLE_DEVICES='' "$PY" -B -m regression_v7.cli \
  --head "$HEAD" --cpu --cpu-module regression_v7.test_gdn
```

A focused test is not a replacement for the required full CPU gate.

## 6. Explicit GPU module or whole-model replay

Only the GPU coordinator should run this section, after the exact-HEAD CPU gate
and required prerequisites pass. The runner preserves other applications,
acquires the common lock, checks the live GPU environment, and bounds its own
child process. It does not authorize killing a desktop process or changing KV,
precision, batch size, or memory settings to force a pass.

Run affected standalone module tests first. Then run `whole` with a different
fresh artifact root. Available module names are:

```text
gemm predicates gdn norm_conv attention_kv gdn_prefill
sampling_runtime pointwise_remaining recurrent_remaining runtime_remaining
```

`--suite modules` runs all ten groups. This example selects one group; changing
`SUITE` to `whole` uses the whole-model path. Retain the shell variables from the
previous section.

```sh
SUITE=gdn
ARTIFACTS=/home/n/work/TileIR/v7-regression-artifacts
R="$ARTIFACTS/manual-${SUITE}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
test -d "$ARTIFACTS" || exit 1
test ! -e "$R" || exit 1

"$PY" -B -m regression_v7.cli --head "$HEAD" --suite "$SUITE" \
  --prepare --artifact-root "$R" || exit 1
"$PY" -B -m regression_v7.cli --head "$HEAD" --suite "$SUITE" \
  --prepare-cache --artifact-root "$R" || exit 1
```

`--prepare` creates metadata only. `--prepare-cache` verifies and copies the
pinned source cache into separate-inode files at `$R/cache_v1`; neither starts
GPU work. Never reuse an old result directory or overwrite a partial copy.

The launch block below preserves the **actual** outer return code even on a
failure. Use it as one shell block; do not manufacture an exit receipt later.

```sh
(
  set -C
  set +e
  "$PY" -B -m regression_v7.cli --head "$HEAD" --suite "$SUITE" \
    --run-gpu --artifact-root "$R" > "$R/OUTER.stdout" 2> "$R/OUTER.stderr"
  rc=$?
  printf 'EXIT=%s\n' "$rc" > "$R/OUTER-EXIT.txt" || exit 1
  if [ "$rc" -ne 0 ]; then
    printf 'HOLD: launcher exit %s; preserve %s and inspect owned cleanup.\n' "$rc" "$R" >&2
    exit "$rc"
  fi
  "$PY" -B -m regression_v7.recovery --head "$HEAD" --artifact-root "$R" \
    > "$R/RECOVERY.stdout" 2> "$R/RECOVERY.stderr" || exit 1
  if [ "$SUITE" = whole ]; then
    "$PY" -B -m regression_v7.cli --head "$HEAD" --suite whole \
      --analyze --artifact-root "$R" > "$R/ANALYSIS.stdout" 2> "$R/ANALYSIS.stderr" || exit 1
  fi
)
```

Do not call internal `--child` directly, fabricate lock/admission environment
variables, retry the same root, or clear a cache to remove failure evidence.
The independent recovery command records a successful run's observed exits and
post-run state; it does not turn a failed run into PASS. After a failed launch,
inspect `regression01-launch.json` and owned cleanup state before any new attempt.
The launcher owns bounded cleanup of its child group, not unrelated processes.

The GPU child has a 900-second bound and the parent a 930-second bound. OOM,
timeout, source drift, unknown compute attribution, or missing evidence is HOLD,
not a reason to silently change settings. A recovery or analysis failure also
prevents the complete qualification verdict.

## 7. Reading results

```text
$R/
  REGRESSION.json, PINS.json        revision and source admission
  CACHE-PRESEED-01.json            explicit fresh-cache receipt
  OUTER-EXIT.txt                   actual outer shell exit
  regression01-launch.json        child exit and owned cleanup
  regression01/RESULT.json         numerical/endpoint/dispatch result
  RECOVERY.json                   independent post-run verification
  analysis/FINAL.json              whole replay only: final attribution audit
```

Whole runs also retain native/lean traces and logger files. These may contain
addresses and local environment details; do not publish raw files unreviewed.

A complete PASS requires all applicable layers, not just `complete: true` in one
file: source/dependency admission, module or endpoint audit, actual process exits,
restoration/recovery, and whole attribution/logger audit. Preserve CPU failures,
partial results, and errors alongside successes. Dedicated module scope is
listed in `regression_v7/COVERAGE.md`; whole endpoints do not independently test
every intermediate tensor or every helper.

## 8. Changes and performance

For implementation changes, follow this order:

```text
CPU -> affected GPU modules -> whole exact/census/no-cuBLAS + recovery
    -> repeated-request lifecycle -> separate repeated warm wall
```

Original baseline bytes are deliberately immutable. An intentional change needs
a reviewed versioned source/contract manifest; the original runner has no bypass
flag. Keep candidate branches separate and never relabel an old receipt as proof
of new code.

`regression_v7/performance_decision.py` implements a 5% engineering regression
budget. The workflow predeclares six to ten independent matched process pairs;
two to five pairs are screening only. Its conditional approximate 95% interval
and decision rule are not a universal noise model or permission to run. There is
no accepted performance baseline or warm benchmark CLI in the f9 regression
runner. Do not use `--run-gpu` diagnostic timings as a substitute.

## 9. Troubleshooting

| Symptom | Action |
|---|---|
| Wrong HEAD, moved tag, dirty source | Stop; verify the intended checkout and reviewed revision. Do not reset or move a tag to suppress the error. |
| Missing fixture/proof/runtime hash | Restore the authorized exact dependency. A source-only clone is insufficient; do not invent replacement tensors. |
| Ignored/untracked local import | Check the source allowlist and dependency closure. Add intentional support through a reviewed commit, not an import-path bypass. |
| Existing artifact root or partial cache | Preserve it and use a new root only after diagnosis. Never overwrite or retry it automatically. |
| Lock busy or GPU admission HOLD | Coordinate with the current GPU owner. Do not kill GUI/user processes or weaken identity checks. |
| OOM, timeout, nonzero child exit | Preserve logs; verify bounded owned cleanup. Do not shrink KV/model settings or silently retry. |
| Unknown kernel/CUBIN set or analysis failure | Keep the result pending. Source names alone and an old CUBIN audit do not qualify new artifacts. |
| Slow profiled execution | Expected diagnostic overhead is not a benchmark. Use the separate qualified warm workflow. |

## 10. Public sources and licensing

The selected implementation uses public-source techniques and public library
input/output comparisons; it is not a reconstruction of proprietary vendor
machine code. CUDA, cuTile, cuBLAS, vLLM, CUTLASS, model files, and reused public
sources retain their own terms and notices.

Preserve bundled component notices and provenance. Examples include
[runtime metadata's vLLM notice](runtime_metadata_five_v1/LICENSE.vllm),
[the attention adapter's vLLM notice](attention_gate_runtime_integration_v1/VLLM-LICENSE),
[the GDN output source notice](tileir_gdn_chunk_output_v1/LICENSE-FLA-MIT.txt), and
[the public attention arithmetic notice](public_fa2_arithmetic_v1/LICENSE).

There is no root project license in the f9 tree. A CUTLASS notice present only in
the original local workspace is not included in that Git snapshot. Publication
requires a source-by-source license/notice and payload review; this guide does
not grant a blanket redistribution license or declare the archive release-ready.
No vendor binaries, model weights, captured tensors, credentials, or raw address
logs are intended to be added to Git. Public-source inspiration alone is not a
substitute for checking the applicable terms.
