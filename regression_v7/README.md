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
