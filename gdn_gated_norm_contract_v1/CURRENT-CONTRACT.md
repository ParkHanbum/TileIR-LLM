# Next API candidate: GDN gated RMSNorm (source contract, not qualified)

Selected by functional boundary size, NOT profiler duration. GemmaRMSNorm is a different API. Verified success is norm_runtime_qualification_v2/normQual01/RESULT.json, SHA e447d80f8e6abcaa6628c275a7bb01c7bcca73730b5fa5968adf2b824b2facff. The v1 incomplete result is a preserved predecessor configuration failure, not the current status.

## Public call and mathematical contract
qwen_gdn_linear_attn.py constructs RMSNormGated(head_v_dim=128, eps=layer_norm_epsilon=1e-6, group_size=None,norm_before_gate=True,activation=output_gate_type). Pinned config omits output_gate_type, so default is silu; swish is normalized to silu. Actual module attributes must still be recorded.
_output_projection(core_attn_out,z) calls self.norm(core_attn_out,z), flattens last two dimensions, then out_proj. TP1 expected logical shape for both inputs: prefill[1696,32,128], decode[32,32,128], producing same shape, followed by[1696 or32,4096] projection input. Values are expected BF16 in this run; exact runtime dtype, row strides, storage offsets and packed-z layout are not yet observed.

RMSNormGated has weight[128], biasNone. Unlike GemmaRMSNorm there is NO weight+1 and NO residual addition. Let xf,zf,wf be FP32 conversions. v=mean(xf*xf,last128); r=rsqrt(v+eps); y=((xf*r)*wf)*(zf*sigmoid(zf)); output casts once to x dtype. This is normalization BEFORE gate; norm(x*silu(z)) is a different operation. Native implementation and public guarded Triton implement this order, but reduction/exponential rounding may differ. Numerical tolerance and raw equality must be independent outcomes.

## Source/backend scope
CustomOp selected forward may be native or CUDA. forward_native calls shared forward_static; forward_cuda calls public flash_linear_attention.ops.layernorm_guard.rmsnorm_fn. That helper flattens leading dimensions, makes last-stride-nonunit inputs contiguous, uses raw weight, allocates empty_like output and returns original logical shape. Source intent is functional fresh output with no X/Z/W mutation, not an in-place ABI. Need actual alias/mutation capture to prove the installed path.

Saved optimized trace inventory K017 has384 events/request (24layers×16tokens) attributed by generated kernel name/source to gated norm. It is an Inductor-generated fusion, not evidence that eager forward_cuda/guarded Triton is the selected path. No actual tensor snapshot or provider evidence for this boundary was found in the existing packed and Gemma banks.

## Minimal next capture
Reuse norm_runtime_contract_capture_v2 eager public apply_model + forward pre/post observer mechanics in a NEW folder, without Gemma IR dispatch wrapper (this module may bypass IrOp). Target suffix layers.0.linear_attn.norm only; capture one actual prefill and one actual decode. Exclude engine warm/synthetic calls by installing after one unobserved request. Record selected _forward_method identity/source SHA and fixed module attributes eps/group_size/norm_before_gate/activation/bias.

Capture X,Z,W BEFORE the call; Y and X/Z AFTER. Save original dtype/shape/stride/storage_offset/storage capacity/pointers, same-object/storage overlap map, logical byte digests, file SHA, current-stream context. Store W once and recheck final digest. Preserve original return identity. Unexpected shapes or options remain recorded unsupported, never silently reshape. Require unobserved/observed512IDs and optimized reference token comparison, but label eager diagnostic explicitly. No vendor/internal instrumentation needed.

## Small implementation after actual contract confirmation
New independent API gated_rms_norm(x,z,weight,epsilon): inference-only explicit functional output; narrow actual shapes/options; unsupported options fail explicitly or qualified original fallback before launch. One row128 FP32 reduction with actual source strides and final BF16 cast, using public cuda.tile. No graph, timing, fusion with out_proj, or cache changes. Keep source Apache-2.0/MIT provenance (vLLM layernorm + Tri Dao FLA guarded implementation), preserve notices if deriving code.

Qualification: rebuild original positive-stride descriptors from saved logical snapshots, compare against exact observed original output and selected public callable plus explicit FP32 reference. Check X/Z/W preservation, fresh output/nonalias, finite inputs, zeros, scale extremes, cancellation, and supported stride/offset tails. Do not advertise arbitrary group_size, bias, missing z, sigmoid activation or norm_before_gate=False until explicitly implemented/tested. Whole runtime replacement and performance are separate, later tasks.

## Why not convolution or RoPE next
Convolution exposes mutable conv state, prefill/decode APIs, packed sequence metadata and activation behavior; existing B-specific batched convolution qualification does not prove vLLM API/state compatibility. QK norm/RoPE is fused with head-interleaved gating, partial rotary64, MRoPE positions and cache-related layouts. Both remain necessary inventory items, but require wider contract capture than this two-input, one-output headwise normalization.

No GPU query/init/lock/compile/capture was performed. No implementation is claimed completed; capture is the remaining fact needed before fixing supported stride/dtype scope.

Numerical distinction: forward_native uses PyTorch FP32 ops with their own rounding; forward_cuda uses guarded FLA Triton sigmoid/reduction and compiler contraction. The new TileIR draft uses ct.sum, ct.rsqrt and 1/(1+ct.exp(-z)), then one BF16 cast. Identical algebra does not imply identical approximation, FMA contraction or raw bits. No generated code was inspected/compiled. Capture selected callable and compare tolerance/raw bits separately.
