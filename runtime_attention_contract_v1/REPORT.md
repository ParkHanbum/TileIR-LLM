# Runtime and eight full-attention segments: functional contract comparison
No model optimization proposed or executed. CPU public-source inspection only.

## Evidence boundary
Observed earlier: same32 copies of one53-token prompt, BF16, greedy16 outputs each,512 IDs equal across vLLM and B; B full81 output/cache gate passes against its own N reference. Existing Nsight showed graph execution and kernel names/counts. Those facts do NOT prove intermediate vLLM/B logits, attention cache bytes, request mapping, arbitrary prompts or all public cuBLAS contracts are equal. Exact vLLM A/B/C pointer-stride/API traces were not acquired here. Following shapes are logical source/config contracts, not newly observed tensor snapshots.

## Common logical flow
Token IDs→embedding→per-layer input RMSNorm→projection→attention-specific normalization/RoPE/cache/attention/output gate→output projection→residual→post-attention RMSNorm→MLP→residual; final RMSNorm→selected-token LM head→logits processors→greedy argmax→append/update stopping state.32 layers, full attention at indices3,7,11,15,19,23,27,31; remaining24 are recurrent linear attention and covered separately.

Pinned config: hidden2560, MLP9216, vocabulary248320, full-attention16 query heads/4 KV heads/head256, q/k RMSNorm epsilon1e-6, partial rotary64 dimensions (0.25×256), interleaved MRoPE sections[11,11,10], theta1e7. No attention bias, inference dropout0. BF16 hidden/projection storage; normalization/softmax intermediate precision is implementation-specific and must be checked, not inferred solely from storage dtype.

## Token layout and positions
B uses rectangular IDs[32,53], then32×1 incremental input, logical GEMM rows1696 prefill/32 decode. HF generation retains request-row order and growing sequence IDs. Public vLLM runner packs scheduled tokens with query_start_loc, positions, slot_mapping and block tables; decode scheduling/graph padding can differ from logical live-token count. For this no-prefix-cache/no-chunked-prefill run,1696 and32 are logical scheduled counts, not proof every kernel sees exactly those rows. Equal repeated prompts cannot expose request permutation.

Text-only prompt positions should correspond0..52;15 incremental forwards consume positions53..67 and leave cache length68, while output sequences contain53+16=69 IDs. The16th selected token has not been forwarded. MRoPE and position-delta generation differ by runtime representation. Must compare canonical request/token position identities, not blindly compare packed positions shape.

## Full-attention operations
B/HF: q_proj produces[32,T,8192], reshape to[32,T,16,512], split each head into query256 and gate256. k/v projections each produce[32,T,1024]→4×256. Q/K head RMSNorm precedes partial RoPE; V and gate do not receive RoPE. Q/K transpose into attention layout, then cache update precedes attention. Output4096 is multiplied by sigmoid(gate) then projected4096→2560.

vLLM: one TP1 packed QKV linear produces token-major[N,10240], split[q_gate8192,k1024,v1024] at row offsets0/8192/9216. q_gate remains head-interleaved Q256/gate256, NOT allQ4096 followed by allgate4096. Public Qwen3NextAttention uses the same conceptual norm→RoPE→attention→sigmoid gate→out order; it may fuse split/QK norm/RoPE/gate-copy depending on source eligibility. Fusion selection and intermediate rounding require trace/snapshot evidence. Projection worker independently confirmed the layout and will provide detailed GEMM contracts.

B nominal attention Q[32,16,T,256], K/V[32,4,L,256]; L53prefill and54..68decode. In canonical dense cache storage, strides are(4*L*256,L*256,256,1). Transpose-derived Q/K views and product contiguous adapters can alter strides; exact per-call snapshot required. vLLM packed projection views and paged KV backend use different strides and physical storage, so source logical dimensions are insufficient to claim byte-identical buffers.

## Cache, mask, ownership and stream contracts
HF DynamicCache owns per-layer contiguous growing K/V created by torch.cat. Recurrent/conv state ownership is separate. vLLM attention receives per-request block-table/slot-mapping metadata and writes into engine-owned paged KV pools; valid tokens, block reuse and sequence-length mapping are part of runtime correctness, not GEMM semantics. Compare logical request/head/position KV values, not whole pool bytes or allocated tail.

TileGym wrapper explicitly discards attention_mask. Its decode implementation uses V.shape[2] as valid sequence length; this matches only an appropriate unpadded dynamic-prefix contract. General padding/mixed lengths/custom masks have NOT been validated. Enabling HF StaticCache without adapting validity is unsafe: full capacity returned while mask ignored; zero unused K/V still changes softmax denominator. kv_len_per_split is split granularity, not valid length. StaticCache supports hybrid construction but that does not establish compatibility with current TileGym consumption.

B ports use the current CUDA device/stream; API success means enqueue, with explicit synchronization at measurement/output boundaries. vLLM uses scheduled streams/events and graph replay; its stream synchronization and buffer reuse lifetimes differ. Contract comparison must enforce producer-consumer ordering and buffer ownership before/after completion. Graph policy and host reads are functional execution conditions here, not optimization recommendations.

## LM head and sampling
Both have tied embedding/LM-head weights logically[248320,2560]. HF selects logits_to_keep before lm_head; runtime-resolved selected rows must be recorded. vLLM exposes compute_logits on selected hidden states and a separate logits processor/sampler. B casts generated logits to FP32 before processors; vLLM public sampler has greedy argmax paths. BF16/FP32 conversion, EOS/min-new constraints, ties/NaNs and output integer dtype can differ. Equal final IDs on one repeated prompt do not establish logits/raw-byte/sampler equality. Distinguish whole512 selected outputs from480 incremental-forward outputs.

## TileIRBLAS boundary and remaining validation
BLAS scope: Q/gate/K/V and output/MLP/LM-head linear algebra, packing/unpacking contracts, dtype/transpose/leading dimensions/strides, alpha-beta semantics and output aliasing. QK score and probability×V products are mathematical GEMMs, but fused attention also owns softmax, scaling, mask and cache semantics; cannot classify whole attention as cuBLAS-equivalent merely because it contains matrix products.
Non-BLAS scope: token scheduler/packing, embeddings, RMSNorm/RoPE, mask/softmax, gating, paged cache management, stop/argmax, stream/event/graph lifetimes.

Required library validation plan (not executed):
1. Map live tokens to(request ID,position) at every region. Include distinct prompts, permutations, mixed lengths, padding, EOS behavior and repeated requests so accidental interchange is detectable.
2. Save input/output plus dtype/shape/stride/offset/ld/transposes/scalars/stream for each BLAS API boundary; unpack packed QKV canonically. Compare same inputs through direct cuBLAS public API and TileIRBLAS with contract-pass, tolerance-pass and raw-bit-pass separate.
3. At the8 attention layers compare post-QK-norm/RoPE tensors, valid-prefix canonical KV, scaled/masked scores or an independent FP32 attention reference, gated output and projection output. Test length53/54/68 and page-boundary lengths, non-contiguous views, all supported masks.
4. Verify cache reuse does not leak another request; preserve identities/validity through graph replay and asynchronous completion. Compare LM-head logits before/after processors, tie/NaN cases and generated IDs separately.
5. Report coverage by actual contract rows and supported input domain, not percentGPU time alone. Generic padding/static/paged-cache/stride contracts remain unsupported until their tests pass.
