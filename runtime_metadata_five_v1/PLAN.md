# Public metadata five-op CPU candidate
Scope: Qwen3.5-4B B32/P53/G16 BF16 fixed eager workload. 2 int32 subtractions, 1 int32 greater-than-zero, 2 int32 full initializers per request. Public installed vLLM Apache-2.0 source only; no proprietary binary analysis. Source snapshots/hash in SOURCE.json and trace rows in METADATA.json. Independent kernels use explicit unsigned modulo32 subtraction then int32 cast, signed gt, fill -1.

Observed: sub inputs shape32/stride1 with offsets1/0 then0/0, gt int32 shape32/offset0. Full arguments values were NOT captured. Size2048 is source inference from actual mlist224 and MAX=max(1024,mlist_len)*2. Runtime rejects other full shape/value before native/candidate. Batch_ptr CPU-int64 to GPU-int32 copy has both HtoD and cast kernel; remains native and outside this five-op scope. Other HtoD remains unchanged.

Source-owned copies preserve two sub expressions, one gt and two full allocations. Cache branch and cached result object identity are preserved. Common class method and GDN builder class method are guarded; GDN copied globals use an owned conv-metadata function copy. No global torch/Tensor patch. Actual cached callers/census not yet observed with bindings installed.

CPU-only prepared. Run requires independent review, final pins/prerequisite and fresh cache. GPU/TileIR compilation NOT performed. Full model native -> shadow5 -> replacement5 always retains independent native oracle calls for raw comparison (5 each phase); only replacement output feeds downstream. No native0/performance claim. Each input full backing bounded1MiB and metadata preserved, output fresh, first failure latch and entry guards. Runtime fixed512 tokens/public2560 values (not full logits). Unexpected scope/census => HOLD, no automatic retry.

Private GPU edge tests planned5: int32 extrema/modulo-overflow sub2, signed gt1, full2. Small CPU tests do not prove compiled integer semantics. Actual alias/source dispatch/return/cache and GPU behavior remain pending.

Native oracles allocate independent outputs for sub/gt/full and do not mutate original inputs. Replacement returns TileIR output downstream, but still runs native oracle five times, so native0 must not be asserted. This standalone proof does not combine other new helper adapters. Runtime-class cached caller coverage is verified by target census, not assumed from source alone.
