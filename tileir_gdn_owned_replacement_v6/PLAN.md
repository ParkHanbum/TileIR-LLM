# Narrow GDN owned-copy replacement v6, CPU review candidate

Unchanged successful v1 numerical APIs/kernels. New replacement harness only. V5 actual shadow prerequisite is required by exact RESULT/launch/recovery SHA (see prerequisite.py); validates child0, recovery0, process absent/lock available, synthetic6 and actual model1152 passes. The coordinator's first system-Python attempt was rejected before admission/child, while the exact-venv run succeeded; neither attempt is relabeled or repeated here. Prior success artifacts are read-only references, not copied.

Runtime returns actual TileIR-produced zero/b/a output tensors to unchanged core/projection. Target native zeros/contiguous calls are ZERO in replacement arm. The unchanged separate native-reference model request naturally still uses native targets: do not claim zero native calls for the whole process. Prior synthetic edge proof is referenced rather than repeating native synthetic targets in this run.

Proof per replacement operation: zero expected CPU bytes are all0 (positive BF16 zero), every returned element checked. b/a expected bytes extracted from BEFORE full-backing CUDA→CPU uint16 snapshot by exact row64/offset0 or32 layout. Candidate output bytes must equal the CPU extraction, and AFTER full source backing bytes must match BEFORE, including unused opposite half. No BF16 numeric conversion, NaN canonicalization or native target oracle. CPU bytes are ephemeral, never serialized. D2H transfer and CPU byte comparison add diagnostic overhead; NOT a performance test. Fresh output backing disjointness against original input and all retained earlier outputs is enforced. No guard-region canary outside exact allocation.

Support remains BF16 CUDA M32/1696, b/a[M,32] stride[64,1] offsets0/32 with exact M64 backing, no autograd/conj/negative bits; zero[M,32,128]. API rejects unsupported metadata; failure latches and no uncertain-launch retry. Actual byte output is passed downstream, so existing512 tokens/public2560 and lean26 dispatch gates independently check end-to-end impact.

Exact cached-bound source_checked regression retained: same source slice/dedent on both sides, filename/line/module/qualname/signature/fileSHA constraints. 24 actual attention owner cached bindings must match pinned public source. Existing core child/26/472 bindings remain unchanged. No four-helper composition. Existing lean26 lifecycle copied by explicit Python allowlist; new package has no runtime/caches/receipts. Combined cleanup restores owned bindings before base bindings, then external sync/release; actual original/core identity gates retained.

18 exact-venv stdlib tests PASS: full inspected-source/cachedbound path, scope guards, unchanged v1 tests, CPU oracle NaN/Inf/signedzero/subnormal payload cases, candidate-return identity/native0 and nonzero zero-result rejection. No CUDA/torch/vLLM import or GPU execution in preparation. No runtime artifacts or cache receipts copied. Existing historical dependency PINS metadata may include cache references but their contents are not opened/rehashed during CPU freeze.

Future command after independent review and coordinator cache/admission preparation:
/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python -B /home/n/work/TileIR/public-tileir-blas-20261111/tileir_gdn_owned_replacement_v6/launch.py --run-gpu --run-name gdnOwnedReplace01

Cache provenance prerequisite intentionally remains pending. No automatic retry, no adoption/performance claim, no claim of100% model coverage.
