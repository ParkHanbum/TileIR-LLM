# All GEMM + sixteen existing adapters + Top5/ranks/selected-logprobs: prospective composition

Status: CPU implementation only. No combined GPU result exists. All earlier source/results remain unchanged. This is a new correctness run, not a reclassification of separate runs.

## Fixed scope and denominator
- Qwen3.5-4B BF16 TP1, eager/compile NONE/graph NONE, B32/P53/G16, the same frozen 53-token prompt repeated 32 times. Original True/True BF16 reduced-reduction/split-K flags read and preserved. Runtime, KV allocation, scheduler and generation settings unchanged.
- Existing 16 adapters: 3,184 selected logical calls. GEMM: 153 selected owners × 16 = 2,448 calls / 13 contracts, including ba K128 360 and unchanged backend 2,088. Top5, ranks and selected-logprobs: 16 enclosing helpers / 48 logical suboperations.
- New prospective total: **5,680 selected logical operations, 5,648 enclosing helper/GEMM boundaries, 18 controllers**. None of those is a GPU-launch denominator or total model coverage.
- Current scoped attention96, all GDN prefill candidates are deliberately absent. The passed ordered selected-logprob v2 is now included. Native cat/output construction, attention core, GDN prefill and uncensused runner/device work remain. See the separately pinned eager_remaining_inventory_audit_v1 snapshot. It is not an exhaustive current eager event inventory.

## Source-backed shared-hook / alias plan
1. Preserve existing16 installation order exactly: norm, attention_gate, block_gather, slot_mapping, pos_seq, gated_norm, sample_counts, qk_rope, silu_mul, indexed_prefill_conv, indexed_decode_conv, post_conv, embedding, attention_kv, greedy_sample, gdn_decode_core. Append all_gemm, then sampler_three. Close in the exact reverse order, even on partial install or component close failure.
2. Sampler.__call__ is shared. Counts installs first, greedy later, Top5/ranks/selected-logprobs last. The outer Top5 wrapper calls greedy -> counts -> original. They replace distinct helpers get_num_sampled_and_rejected, gumbel_sample, compute_topk_scores, each exactly once per original sampler call. Each owns a separate pending context. CPU tests execute the real three install_classes bodies with mock public helpers, including body exception and identity restoration.
3. GPUModelRunner.prepare_attn is shared by gather then slots. The existing two-wrapper chain is unchanged; reverse-close slots before gather. prepare_inputs/pos-seq is a distinct hook.
4. GDN prefill/decode convolution share causal_conv1d_fn. Existing prefill-first, decode-second delegation is retained. GDN decode core installs after convolution/post-conv and records their then-current bindings as untouched. It must close before those bindings are restored. Its own core helper is distinct. No flattening or replacement of this chain.
5. GEMM uses per-instance quant_method.apply. TP1 embedding replaces the distinct quant_method.embedding attribute. All178 quant-method instances are distinct in the pinned original census;153 apply targets and25 guarded non-GEMM applies are retained. Embedding may use the same owner object as an excluded apply guard, without changing its actual embedding method. No class-wide GEMM monkey patch.
6. Attention gate replaces instance.forward but calls live qkv_proj/o_proj; QK/RoPE uses _project_qkv_gate/helper. GEMM therefore reaches the same instance apply dispatch within the public forward. Silu hooks instance.op, norm/gated norm hook their own forwards. No direct target overwrite was found in this CPU source audit; combined live proof is still required.
7. Native-shadow returns native values at each supported functional boundary; mutating convolution/KV/metadata/GDN candidates use the existing scratch/state proof. Replacement returns the qualified alias/tuple/layout contract. GEMM records fresh output immediately; later legal consumer mutation of a packed-output view is not a GEMM-output preservation violation. Do not claim untouched full state when the component only proves selected state or compact remapping.
8. GEMM weights retain four full-byte phase-boundary scans and each-call owner/storage/layout/version checks; no per-call W-byte claim. All2,448 X/Y raw/strict/finite and cross-phase SHA gates remain. Embedding's selected-row/version proof remains separate.

## Failure and lifecycle
- All three component groups must pass full shadow in this combined request before any replacement request.
- Existing all-GEMM and Top5 first-failure latches remain. Three older wrappers (norm, gated_norm, decode-conv) formerly recorded a failed shadow then returned native until finish. The new controller wraps only their diagnostic save callbacks: write original failure evidence first, then latch and raise. This adds no arithmetic or public target wrapper and avoids breaking GDN untouched-binding assertions. Save callbacks are restored before adapter close.
- Exceptions propagate to the existing bounded cleanup; there is no automatic retry. Partially installed modules are included in reverse cleanup, and all earlier closes are attempted if one close fails. A failure makes the result incomplete even if some independent work already executed.
- OOM or timeout is HOLD with original exception and existing partial evidence. No reduced KV setting, changed batch, alternate math mode, fallback or retry. New source only; no old cache/run/receipt files copied.
- Fresh cache receipt must name this exact cache_v1 destination, match source/destination file hashes and distinct inodes, and pass before admission/child creation and again before torch/vLLM import. Launcher/worker receipt SHA must match. The earlier Top5-only cache exception remains historical and unchanged. This v2 uses the newer passed three-way sampler v2 with a valid cache proof; its receipt is not copied or reused as this run's receipt.

## Memory risk and measurement limits
- Separate previous allGemm shadow: allocated14,347,312,128 B / reserved14,417,920,000 B.
- Separate sixteen-adapter shadow: allocated14,623,536,640 B / reserved15,114,174,464 B.
- Separate Top5/ranks/selected-logprobs shadow: allocated14,314,281,984 B / reserved14,386,462,720 B (samplerThreeOrdered02).
- These peaks cannot be added or used to prove combined fit. Persistent convolution/GDN scratch and differing allocator history may exceed RTX4080 capacity. Fresh admission does not prove future peak fit. GPU coordinator must preserve GUI and record an OOM rather than changing the workload.
- Existing 900s child / 930s parent bounds retained. Correctness hashing and diagnostics dominate this run; internal wall fields are not performance comparisons. GUI activity is not controlled as a performance experiment.
- CPU tests of projected separate result records are schema-only fixtures, never evidence of a successful combined GPU run. Only a fresh actual RESULT with actual exit, matching SHA, independent recovery and unchanged pins can establish this composition.

## Version v2 delta

Preserved v1 was CPU-frozen and never run. This new folder imports sampler_three_ordered_runtime_integration_v2 in place of top5_ranks_runtime_integration_v1. It reuses already-qualified Top5/ranks/ordered-score numerical code unchanged and replaces the prerequisite result and auditor. Selected logical operations rise by16 to5680; boundary count remains5648. No attention96 or GDN prefill added.
