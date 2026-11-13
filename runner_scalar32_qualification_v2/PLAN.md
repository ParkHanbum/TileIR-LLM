# Scalar32 runtime qualification v2

CPU source preparation only. Original scalar API change is solely `from kernel import fill_one` → explicit package import. kernel.py arithmetic unchanged. Source-only allowlist; no runtime/cache/receipt copies. Metadata sub/gt/full candidates remain separate and are not installed.

Worker reuses established native eager B32/P53/G16 T/T public vLLM request settings and frozen512 token/public2560 reference. It runs native baseline, private32 synthetic original-fill vs TileIR full128B proofs, model shadow32 then conditional replace32. Other model operations remain native. Timing/profiling absent. Replacement requires exact corresponding full preimage/index order from shadow and output backing bytes, all32 distinct indices, exact endpoint gates.

Actual typed model_state tensor is admitted only for cuda:0/int32/[32]/stride1/offset0/exact128B/alignment4, max_num_reqs32, alignfalse, recoverSSM absent and speculation absent. Identity/pointer/extent guards before super and target entry; no address-to-tensor reconstruction. Source-owned add_request changes only original scalar fill expression and retains lexical __class__/super order. Instance-cached add_request is rejected. Actual runner.add_requests is checked against original file/path/lineno/exact slice→dedent AST; its source calls self.model_state.add_request dynamically. Actual32 dispatch census is still required, not inferred from source.

Launcher: CPU dry by default, PINS then destination-bound fresh-cache gate before GPU admission, shared LOCK_NB, exact existing admission loader, isolated child process and900s worker alarm/930s parent timeout, owned process-group cleanup, actual exit/result hash, post-admission and lock recovery. No admission/GPU/cache/compile was performed by this worker. Parent supplies fresh cache after independent source review. Preserve settings on failure; no automatic retry.

CPU16 tests and run/launch dry succeeded with actual venv. Tests use no real Torch/CUDA imports: fresh subprocess stops before framework imports; mocked import traversal with poisoned generic kernel proves qualified kernel lookup; real-file AST fixtures validate super/caller and lifecycle. Numerical GPU behavior, current actual owner/path, outputs and performance remain unverified.

Full source PINS and CPU-READY freeze are now completed. Do not execute until independent review and parent fresh cache/admission conditions are satisfied. Prior v1 and public draft remain unchanged.
