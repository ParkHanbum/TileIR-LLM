# ORIGINAL v7 executable regression

This is a separate test commit on immutable baseline
`0b9611661ef216c35fb9f4c5da92662e5930dcbb`, tag
`v7-own-kernel-fixed-workload-20261113`. No candidate G/H/O/M, warm
measurement, or performance assertions are part of this runner.

Use the installed pinned Python 3.14 venv and `-B`. All commands require
`--head <exact regression commit SHA>`; the baseline tag must still point to
the baseline commit. Tracked files must match HEAD, original allowlist bytes
must be unchanged, and ignored/untracked local imports or code execution
are rejected. Existing unrelated ignored artifacts may remain on disk.
This is deliberately an environment-bound replay, not a standalone download.
External runtime/model/evidence pins and per-module data pins are checked.

## Commands (coordinator only for GPU)

```
python -B -m regression_v7.cli --head HEAD_SHA --cpu
python -B -m regression_v7.cli --head HEAD_SHA --prepare --suite whole \
  --artifact-root /home/n/work/TileIR/v7-regression-artifacts/uniqueWhole
python -B -m regression_v7.cli --head HEAD_SHA --prepare-cache --suite whole \
  --artifact-root /home/n/work/TileIR/v7-regression-artifacts/uniqueWhole
# Explicit pinned 1033-file copy only; no GPU or legacy cache mutation.
R=/home/n/work/TileIR/v7-regression-artifacts/uniqueWhole
set -C  # refuse shell output overwrite
set +e  # preserve the actual exit even when the command fails
python -B -m regression_v7.cli --head HEAD_SHA --run-gpu --suite whole \
  --artifact-root "$R" > "$R/OUTER.stdout" 2> "$R/OUTER.stderr"
rc=$?
printf 'EXIT=%s\n' "$rc" > "$R/OUTER-EXIT.txt"
# Do not create a successful exit record by hand or retry a failed root.
python -B -m regression_v7.recovery --head HEAD_SHA --artifact-root "$R"
python -B -m regression_v7.cli --head HEAD_SHA --analyze --suite whole \
  --artifact-root "$R"
```

For representative numerical GPU tests use `--suite modules`, or individually
`gemm`, `gdn`, `predicates`, with a distinct fresh artifact root/cache each time.
The common lock, installed interpreter, GUI-preserving admission, bounded child,
process-group cleanup and post-run recovery are reused from the proven launcher.
No retries or option/KV/precision changes. CPU modules run in separate processes.
The original workload remains B32/P53/G16, BF16, T/T, eager mode0/graphsNONE.

Whole replay preserves the original request function AST exactly, and the
original engine/install/restore lifecycle. Only artifact-root qualification,
child entry and standalone process cleanup imports are adapted. The native
and lean request each emit an actual trace and official library logs; native
classic GEMM2448 and lean0 plus cuBLASLt positive-control checks are required.
The explicit destination root is passed into the original result audit.
The fresh own cache analyzer then requires 13,300/13,300 correlated lean compute
kernels matched by exact ELF symbol, bytecode and pinned public source; D2D and
memset must be zero. CPU operations, sync and all transfer categories remain in
the saved trace ledger. Names alone are not proof. Loaded-module SHA remains
unobserved; no claim of all APIs/all GPU work or performance parity is made.

`--prepare` creates only source/admission metadata, not a cache and not a GPU
process. Cache preseed and GPU execution are never automatic from CPU tests.
Existing run paths cannot be overwritten. A failed child or recovery fails the
launcher; an incomplete source-symbol join fails final analysis. Historical
proof is admission evidence, never a substitute for the fresh regression.

This initial runner is **baseline qualification**, not yet a permissive future
implementation-change regression policy: every original baseline byte remains
fixed. A future intentional implementation change requires an explicit reviewed
versioned manifest/contract update; there is no bypass flag. The baseline tag
itself remains immutable.

`--prepare-cache` verifies the pinned historical source manifest and all 1,033
source bytes, makes new separate-inode files, and runs the original cache gate.
A partial failed copy is preserved and not retried or overwritten; choose a new
artifact root after diagnosis. It never requests GPU admission.

Final analysis also compares the 172 own CUBIN SHA set to the pinned three-process
internal-call audit. Exact equality reuses its 65 local-call-site proof. Any new
or missing artifact is saved as pending and fails baseline final PASS; no new
SASS dumping or vendor binary analysis occurs.

## One historical test-phase exception

The preserved original test
`whole_lean_extensions_runtime_v7.test_cpu.ExplicitRootTests.test_launcher_explicit_root_and_source_only`
is reported as **SKIP**, exactly one ID, by the regression loader. Its final two
assertions required no historical cache or preseed receipt before the original
experiment. Those artifacts now legitimately exist and are preserved. Its useful
`audit(result, root=R)` AST assertions execute in
`regression_v7.test_runner.Gates.test_original_launcher_explicit_root_port`.
No `Path.exists` is mocked, no old cache is deleted, and no other test is skipped.
The legacy v2 circular-result negative fixture is separately tracked as test
input; it is not a production adapter. The first failed exact-HEAD CPU run is
preserved. No GPU qualification is allowed before the corrected full CPU catalog
passes at its new committed HEAD.

Supplemental test import closure: the previously empty packed_grouped_qualification_v1 package initializer and original tileir_cat_v1 pins/kernel definitions are tracked by a test-support commit. They do not change the active v7 implementation or move its baseline tag. Exact-HEAD admission first caught the missing package initializer; the failed CPU receipt is preserved.

## Exact post-update owned-code provenance

The first whole regression failure is preserved: the import/exec policy rejected
`residual_four_gdn_owned_lean_v2/four.py.post_update_guard_copy` during lean
installation, before candidate execution. This is a known **local**, not external,
virtual filename produced by the baseline public post-update binding.

One exact exception now requires the tracked `public_post_update_guard_v1.binding`
`make_method` frame/code, the tracked original `four.Replacement.call` code and
module globals, the exact virtual filename, and full AST/compiled-code equality
with the single approved `validate_values(torch,args)` to
`_post_update_value_guard(torch,args)` substitution. Unknown virtual paths,
foreign generators, altered source/code/templates remain rejected. No generic
suffix/prefix or ignored-file allowance exists. Reconstruction only compiles CPU
Python ASTs without executing them; it cannot recurse through the exec audit.

The CPU regression executes the actual v7 `Extension.install_post_guard` through
`Checks`, production `Binding.install`, `make_method`, and reverse close with
import and exec guards active and framework imports prohibited. Seven negative
provenance cases are rejected. This is construction/lifetime proof, not a GPU
or endpoint PASS; a new exact-HEAD whole replay remains required.

## Preserved lazy GEMM facade dependencies

A subsequent whole replay reached lean execution, then failed before completing
the first request because the tracked-only policy found the original
`tileirblas_decode.api -> tileirblas_tuned` dependency outside the initial source
allowlist. Both original tuned facade files are added unchanged in a supplemental
test-support commit. Their SHA256 values match the historical decode baseline:

- `tileirblas_tuned/__init__.py`: `e64a72719397c5d28d686e0ac1d3ca8237c525b48535742e516a6a8718832914`
- `tileirblas_tuned/api.py`: `e333e39e9d1f1650dfc82911d31e13fc91fe808895b29b9128f780822a8087ab`

No numeric source or dispatch behavior is edited, and the immutable baseline
tag is not moved. The failed run and its independent recovery remain preserved.
`test_lazy_imports` statically checks all four literal kernel-module targets,
parent package initializers and their 15-file local import closure against HEAD,
including an intentional missing-initializer negative. This is not a proof of
arbitrary computed dynamic imports; the actual execution guard remains active.

## Bytecode source-table analysis regression

The first completed whole request and successful recovery are preserved even
though its initial CPU analysis failed on an undelimited bytecode string table.
The parser now bounds each filename candidate at the next repository-path
anchor, requires its exact pinned path before reading, and verifies source
bytes. Four CPU cases cover directory/symbol concatenation, `.pydefault`
string-table suffixes, multiple paths, and rejection of unpinned source names.
The same saved trace/cache passed full analysis in a separate create-only
output directory; analysis source SHA is recorded separately from runtime HEAD.
This changes postprocessing only. Unknown compute symbols, changed source, or
a changed CUBIN set still fail rather than becoming a new implicit allowlist.
