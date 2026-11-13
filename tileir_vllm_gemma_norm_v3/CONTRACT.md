# GemmaRMSNorm functional candidate, unqualified

Choice: actual Qwen3.5 imports GemmaRMSNorm as Qwen3_5RMSNorm, rather than generic RMSNorm. Public module forward(x,residual=None) is the target. Signature here: gemma_rms_norm(x,raw_weight,epsilon,residual=None). No-residual Tensor; with residual returns (normalized,new_residual). Both fresh, no input mutation, current stream, no vendor fallback. Not registered into model yet, not the _C.fused_add_rms_norm in-place API. X/R aliases are permitted since read-only; no supplied-output API.

Narrow scope: 2D CUDA BF16 X/R, M32 or1696 and width2560; innerstride1,rowstride>=2560, nonnegative offset. Raw learned W is contiguous2560 BF16/FP32, any nonnegative offset, same device. Positive finite epsilon, no bool; autograd unsupported. Whole metadata validated before allocation/launch. Nonstandard shapes, weightless/variance_size, dtype, quantized/fused QK-RoPE excluded.

Semantics from vLLM0.32.0 public source: promote raw W to FP32 and add1; X and residual promoted then added in FP32; return residual cast BF16, but normalize using unrounded FP32 sum. Sum square/2560, rsqrt(variance+eps), multiply effective FP32 weight, only then BF16 cast. Actual reduction/rsqrt/FP contraction can differ, so source-level contract is not raw or tolerance qualification.

Existing optimized inventory residual_norm has6144 kernel events over6requests,1024/request,total22.185024ms. These are two generated Triton names, only source/name attribution, not an exact observed module/API mapping. Eager backend may differ. No supported fraction or performance gain is asserted.

Required next: original module boundary capture, before and after X/R/W, dtype/strides/offset/alias and epsilon. Suggested layer0.input_layernorm no-residual prefill/decode and layer0.post_attention_layernorm residual prefill/decode, actual signature must be observed. Compare same saved inputs against actual original API outputs, residual and normalized output separately; raw bit/tolerance/finite/in-place alias contract separated. Unsupported alias semantics may require a distinct API, not silently implemented inplace. Existing bank inputs do not establish norm inputs.

Status CPU-only source/mocks, no cuda.tile import/compile/GPU execution. No runtime integration or support claim. Source loop expression independent of proprietary internals; MIT/NVIDIA TileGym kernels not copied.

V2: explicit public ATen vec4 reduction order, RN square/add, mean multiply. Uninstrumented sibling of norm_aten_order_v1. Both HasResidual branches; same validation and allocations. Not yet globally qualified. Prior v1 four cases remain historical evidence only; no claims of v2 equivalence before runtime shadow.

V3 replaces v2 single kernel with square FP32 storage, mean FP32 storage, output. No dummy diagnostic stores, no private flags. Same width/shape/dtype/strides/alias/stream/inference contract and current-stream allocations. Source-only pending actual qualification. Public installed cuda.tile compiler/stub and tileiras public --help expose no contraction switch found; explicit RN previously contracted in actual own PTX, so not relied upon for these two boundaries. Does not guarantee all remaining lowering/rsqrt equivalence.
