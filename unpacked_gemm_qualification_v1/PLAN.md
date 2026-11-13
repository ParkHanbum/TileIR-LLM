# Actual five-contract, eight-bank qualification

CPU preparation only; GPU execution requires independent review and the parent-owned serial window. This is an isolated correctness test of the unchanged public `tileirblas_once_occ2.linear`, not a new GEMM algorithm and not a model replacement or performance test.

Source evidence is the completed native-only v2 capture. Five independent groups: out prefill (GDN and attention banks), out decode (GDN and attention banks), down prefill, down decode, LM head (prefill/first-decode banks). A numerical/finite mismatch halts that group at its first failed call and skips remaining calls/banks in that group; all other groups still run. Unexpected API/compile/CUDA errors or storage mutation abort the entire process, because continuation would not be safe. Numerical HOLD is diagnostic completion, not adoption.

Each bank loads exact saved X/Y and streams its checkpoint named weight in <=16MiB chunks. Full named weight SHA must match the captured live/checkpoint/final SHA. No new persistent weights are written. X/W/Y use exactly the captured dense shape/stride/offset/storage span. These captures have no holes. Whole X/W bytes are rehashed after every call, and output identity/layout/nonaliasing are checked. Input storage mutation is fatal. Maximum live mathematical tensors are under 2GiB; CUDA allocator/runtime overhead is extra and actual peak is recorded.

A fresh independent public cuBLAS handle runs native GemmEx99 twice with math0, compute68, HOST alpha1/beta0, BF16 T/N. Both must reproduce saved native Y raw and strict and finite before any TileIR call in that bank. Captured pointer mode/alpha-beta bytes/internal algorithm were not observed; scalar values are source-derived and replay uses HOST explicitly. Matching saved Y is therefore essential rather than assuming handle equivalence.

The existing TileIR API then executes twice into a preallocated, NaN-poisoned output. Each must exactly reproduce saved Y using unchanged raw-bit and 1e-5+1e-5*abs(ref) strict gates plus finite checks. First differences and one bounded full failed output per group are retained (aggregate logical failure cap64MiB). At most32 logical calls. Original True/True Torch flags are only read and remain unchanged. No per-channel/data correction or tolerance relaxation.

Worker570s/parent600s, compile timeout90s, per-run caches only. No auto retry. The existing reference handle/cleanup/admission/common-lock launcher is reused. The parent verifies actual exit0 (all sampled contracts pass), exit2 (completed numerical HOLD), or exit1 (incomplete/error), then independently checks process absence/lock/recovery. A sample pass does not prove every original call or model-level replacement.

CPU dry entry: `python3 -B launch.py --manifest MANIFEST.json --manifest-sha <sha> --run-name fiveGemmQual01`. Parent adds --run-gpu only after review. The worker is not launched in CPU preparation.
