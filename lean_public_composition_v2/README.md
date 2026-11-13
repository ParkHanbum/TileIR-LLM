# Lean25 CPU draft

This separate package is not installed, GPU-qualified or timed. Existing `lean_public_composition_v1` and all numerical APIs are unchanged. Public source copies retain their original referenced provenance/licenses. The 17 unchanged component classes are imported by reference; SamplerThree is replaced by SamplerFour, and seven new components are appended. No blanket method/data wrapping is used.

Expected request census is 5,888 selected boundaries and 6,056 logical operations, not physical GPU events. Original18 lean was qualified independently; its evidence does not qualify these new wrappers. A numerical25 composition proof also cannot substitute for a new lean25 correctness proof.

## Files
- `sampler.py`: same top5/ranks/ordered-logprob APIs plus public cat; BF16-to-FP32 expression only, retaining original downstream processors and original BF16 ownership.
- `core_parent.py`: public six-stage GDN prefill chain, explicit scoped parent-state four-operation endpoint wrapping the exact installed lean core.
- `attention_runner.py`: attention caller-output endpoint, padding/position pending-context bridge, hidden activation gather and accepted scatter.
- `adapter.py`: exact counts, failure latch, installation order and reverse cleanup. Parent installs last and closes first. Existing Base close retains controller/UVA references until caller synchronizes and calls `release`.

External snapshots, byte hashes, diagnostic oracles and explicit diagnostic synchronization are removed. Public APIs, value-range/finite guards, backend/layout/alias restrictions and necessary D2H checks remain. These guards can still launch native Torch kernels and synchronize. No claim that this is guard-free or 100% TileIR.

CPU tests cover mock control flow only. Required next qualification: same-request fixed512 IDs/public2560 outputs, T/T readonly, exact census, zero target fallback, input/output/state ownership and cleanup, frozen source/cache checks and actual exit/recovery. Timing must separately exclude warmup/compile/diagnostic qualification; any profile must use the same request denominator and classify retained guards, residual compute and unknown callers honestly.
