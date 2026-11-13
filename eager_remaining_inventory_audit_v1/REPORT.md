# Eager Qwen3.5-4B residual inventory, append-only snapshot

B32/P53/G16 BF16 TP1, 32 identical prompts, eager/compile NONE/graph NONE. CPU-only audit of saved results and installed public source. No new GPU, query, compile, lock, cache modification or runtime change.

## Denominators
- **API contract denominator is not closed.** Sixteen adapters are sixteen selected functional boundaries, not all vLLM/ATen APIs and not ABI replacement.
- **3184 is selected logical calls/request** in the completed sixteen-adapter composition. Coverage entries sum exactly3184; RESULT complete=true, SHA b62fcdbc8c20697d304ac0a4340ad0c1820a6cf4a4fd81474b03119716d80eda. It is not total model GPU calls.
- Top5+ranks adds two independently qualified functional boundaries,32 suboperations inside16 helpers. Its model run was separate, selected scores and cat retained native. It has the recorded preseed provenance exception. Do not sum independent proofs into a tested combined run.
- ba decode360 is a subset of packed/all-GEMM2448, not an additional360. All-GEMM is running per parent at this snapshot; no success inferred. Plan reconciles153 owners*16=2448,13 shape/ld contracts. One GEMM API can dispatch multiple physical kernels.
- Attention plan is96 candidate/128 logical attention calls,32 explicit native length65-68 exceptions; not96/128 total GPU coverage. It is a planned runtime experiment, not read here as completed.
- Six GDN prefill stages each have a representative-bank pass reported by parent. They are not a validated chain or24-layer proof; do not infer144 measured calls.
- **Current eager total GPU event count/time share is unknown.** Historical67 names/46980 events over6 requests belongs to a different optimized/graph trace. No numbers from it are used to estimate current remaining fraction. Copies/transport and kernel events also have separate denominators.

## Existing selected composition
|Boundary|Logical calls/request|
|---|---:|
|norm|1040|
|attention_gate|128|
|block_gather|16|
|slot_mapping|16|
|pos_seq|16|
|gated_norm|384|
|sample_counts|16|
|qk_rope|128|
|silu_mul|512|
|indexed_prefill_conv|24|
|indexed_decode_conv|360|
|post_conv|24|
|embedding|16|
|attention_kv|128|
|greedy_sample|16|
|gdn_decode_core|360|

## Additional residual boundaries
“Source path” means installed source establishes the operation and its guard, with no new direct call/event capture. Counts are explicitly unknown unless stated. Live sampler capture proves its enclosing path, not every downstream kernel association. An empty allocation or view is not itself asserted to launch a GPU kernel.

|Boundary|Contract / outstanding support|Evidence and count|Status|Next step|
|---|---|---|---|---|
|sampler selected scores|compute_token_logprobs: BF16[32,248320]+int64[32,6] -> FP32[32,6]|16 native helper subcalls observed in top5+ranks replacement; sample/logprob.py:84-106; top5+ranks RESULT|ACTUAL_RETAINED / RAW_HOLD|Fix exp/reduction difference, full16 raw shadow before replacement|
|sampler ID assembly|torch.cat(sampled[:,None], top5, dim=1): int64[32,1]+[32,5] -> fresh[32,6]|16 source-derived within captured helper16; no separate event census; sample/logprob.py:128-129; top5+ranks adapter copied helper|ACTUAL_HELPER_SOURCE_RETAINED|Implement typed concatenate/output construction with same alias/layout|
|sampler input gathers|positions[logits_indices], input_ids[logits_indices]|source-derived 2 per sampler; actual descriptors/count unknown; sample/sampler.py:forward|SOURCE_PATH_UNQUALIFIED|Capture actual dtype/stride/index and preserve owner; then indexed gather|
|sampler count seed|seq_lens.new_ones(num_reqs), GPU integer tensor passed into already ported count helper|source-derived 1 per sampler; no direct census; sample/sampler.py:forward|SOURCE_PATH_UNQUALIFIED|Typed fill/ones; do not credit count-helper replacement for caller allocation/fill|
|sampling preprocessing|FP32 logits copy/cast and enabled logits processors/temperature/top-k/p masking|actual greedy bank is masked FP32, but each producer/branch/count unresolved; sample/sampler.py:apply_sampling_params/_sample_random; greedy capture|UNKNOWN_SUBCALL_ATTRIBUTION|Capture actual settings and producer associations; do not assume all penalties active or no work because greedy|
|runner no-draft metadata|arange(num_reqs+1,int32), zeros(num_reqs,int32), H2D idx_mapping/query_start_loc|source no-draft path; calls/events not independently captured; model_runner.py:1354-1492|SOURCE_PATH_UNQUALIFIED|Port device arange/fill; retain transport as transport with separate bytes|
|prefill token preparation|prepare_prefill_inputs mutates input_ids/next_prefill_tokens from indexed request state|source conditional has_prefill; expected once here, actual census unknown; input_batch.py:321-341; model_runner.py:1450|SOURCE_PATH_UNQUALIFIED|Capture first actual prefill state/input/mutation contract|
|sampled token combination|combine_sampled_and_draft_tokens mutates input IDs, returns logits_indices; no-draft still uses helper|source every input preparation; exact count unknown; input_batch.py:467-501; model_runner.py:1472|SOURCE_PATH_UNQUALIFIED|Implement observed no-draft branch only, retain unsupported speculative cases|
|hidden-state selection|hidden_states[logits_indices] before LM head, distinct from GEMM|source every sample for non-sharded TP1; actual stride/count unknown; model_runner.py:1595-1620|SOURCE_PATH_UNQUALIFIED|Capture BF16 prefill/decode rows and gather; head GEMM coverage does not include this|
|post-sampling state update|post_update updates computed tokens,last sampled,all-token buffer,total_len; optional histogram|source actual sample_tokens path; count/optional branch unbound; model_runner.py:1679-1707,2217; input_batch.py:post_update|SOURCE_PATH_UNQUALIFIED|Capture exact mutation and alias contract; model_state.postprocess_state separately inspect|
|other request-state maintenance|post_update_num_computed_tokens, model_state.postprocess_state, request-add draft_tokens.zero_, conditional KV block zero/copy|conditional paths; no actual per-branch census; model_runner.py:1254-1270,2329; states.py:117|UNKNOWN_CONDITIONAL|Separate setup/request admission/steady step; exclude inactive spec/PP/TP branches|
|GDN prefill state movement|ssm_state[prefill_state_indices]; mask/invert/zero initial state; indexed final-state assignment|24 prefill core callers known, individual movement counts not captured; qwen_gdn_linear_attn.py:1590-1619|SOURCE_AROUND_ACTUAL_CORE_UNQUALIFIED|Port gather/masked init/scatter around six-stage chain; preserve full selected state and untouched bytes|
|GDN other movement|conditional contiguous/casts/output copies and mixed prefill/decode stitch cat|not all branches active; exact invocation/identity vs allocation unknown; qwen_gdn_linear_attn.py:1622-1626,1978-1979|UNKNOWN_CONDITIONAL|Bind actual eager branch; never count reshape/view as GPU work without a launch|
|async output transport|sampled IDs/counts/logprobs .to(cpu,non_blocking=True), event/stream synchronization|source actual output path; bytes/event counts unknown in eager target; async_utils.py:115-174,264-265|TRANSPORT_NOT_TILEIR_COMPUTE|Measure separately; device kernels cannot replace host transfer/event semantics|
|remaining generated/ATen operations|all uncensused caller-to-device operations including allocator-induced fill/copy|unknown; old optimized trace is not eager denominator; vllm_library_inventory_v1 historical only|UNKNOWN_GLOBAL_COMPLETENESS|One supported eager occurrence inventory needed after composition, excluding diagnostic comparisons/snapshots|

## Completion plan and exclusions
1. Finish existing all-GEMM/attention/scalar-score investigations without duplicate runs. Integrate GDN six-stage chain plus indexed state movement and qualify all24 actual layers. Extend attention to65-68 only with independent raw proof.
2. Capture the small remaining runner/sampler boundaries above in one bounded native observation: caller association, actual shape/dtype/stride/index/value/mutation, exact count and inactive guards. Existing source-only expected counts are not promotion to observed counts.
3. Implement typed gather/cat/fill/arange and request-state helpers with alias/sentinel/mutation gates. Reuse prior kernels only after matching actual contracts; older TileGym copy ports do not establish vLLM integration.
4. Combine successful adapters in a new conditional shadow/replacement run. Then obtain same-run eager event/caller accounting outside correctness snapshot/compare work, with native fallbacks explicitly listed, to close the GPU-event and time denominators. H2D/D2H transport and runtime control remain separate from compute coverage.

Not requested/inactive in this workload: general stochastic sampling, speculative decoding, mixed/chunked long prefill, multi-GPU collectives, multimodal branches, arbitrary batch/shape/dtype, quantization. Their code presence is not evidence of an actual remaining launch. Setup/model load/warmup has a different boundary from a measured request. Source guards such as optional masking, MOE padding, request cache zero/copy require observation before declaring active. This audit identifies real retained boundaries and specific source-derived gaps, but cannot establish an exhaustive current eager GPU library inventory or a percent complete.
