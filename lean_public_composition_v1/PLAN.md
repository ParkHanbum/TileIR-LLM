# Lean public composition v1: CPU implementation, GPU qualification pending

## Fixed scope
This is a new source-only binding layer for the already qualified combined 18-controller request, not a numerical kernel revision. Qwen3.5-4B, 32 identical 53-token prompts, 16 generated tokens, frozen 512 token IDs and 2,560 public top-5 logprob entries. Retain the original eager runtime, precision True/True, KV configuration, public candidate APIs and supported input domains. The previous combined proof is unchanged.

Expected dispatch is 5,648 selected boundaries and 5,680 logical operations: the previous 16 controllers (3,184), all 13 GEMM contracts (2,448), and Top5/ranks/selected-logprob (48 logical operations inside 16 helpers). The new four-way sampler/cat proof is deliberately NOT incorporated. Attention core, GDN prefill core, native cat/output assembly and remaining runner operations stay native. This is not 100% workload coverage.

## What changed, and what did not
Each lean wrapper directly invokes the exact qualified public API on the original tensors or detached views of the same storage. External diagnostic snapshots, output/raw comparisons, per-call tensor hashes, proof JSON writes, explicit synchronization and shadow calls are omitted. Registry installation and reverse restoration order are fixed. Counts, owner association, descriptors, failure latch and supported-scope guards remain. Public API-internal validation, allocation, launch order, arithmetic, precision and support scope are untouched.

This is **not** a fully validation-free production implementation: small device-value checks needed to enforce the qualified scope, finite-input checks, Python census checks, and API-internal validation may still launch native operations or synchronize. A future profile must separate those from candidate kernels and residual native work. No speedup follows from source inspection.

## Lifecycle
1. Install once on a fully constructed, native-warmed, idle model after independent review and frozen source checks.
2. Start only immediately before the single supported request. Calls outside an active request, unsupported shapes/options, count overflow, or any candidate failure latch an error and reject subsequent selected calls. No fallback or automatic retry.
3. Finish requires every expected owner/phase census, GEMM 360 K128 + 2,088 existing backend calls, 24 delegated TileIR prefill calls, and all total counts.
4. Close all controllers in exact reverse order even on error. Restore inherited and instance-owned attributes without silently overwriting a foreign binding.
5. Keep controller/UVA owners reachable through the model after close. The caller must synchronize the device at the request/cleanup boundary before `release(model)` or dropping the engine. There is no hidden synchronization in the adapter. On cleanup error preserve the error and terminate the disposable model process; do not continue another request.

## Memory / failure policy
No full backing clones, per-call W hashes, shadow outputs or diagnostic caches are retained by these bindings. Live output/state allocations inside the unchanged public APIs remain necessary. UVA owners are retained for asynchronous lifetime safety. This reduces diagnostic work but does not establish that the full runtime fits memory. On CUDA OOM, unsupported input, assertion, mismatch or cleanup error: stop, save the harness failure, release only owned resources safely, record actual exit and independent recovery. Do not lower KV capacity, change flags or retry automatically.

## CPU preparation versus actual proof
CPU mocks verify binding order, restore/error paths, exact argument routing, storage/return identity and census guards. They do not prove CUDA numerical equivalence, runtime source compatibility, asynchronous lifetime safety or performance. The old raw proof applies to the unchanged public APIs and old wrappers, not automatically to this new composition. New native-versus-lean model correctness remains mandatory before profiling.
