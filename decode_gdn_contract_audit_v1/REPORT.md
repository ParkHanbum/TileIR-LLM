# Decode GDN/conv/state contract comparison

CPU-only public-source and saved-trace audit. No new optimization code, compile or GPU run. Scope is Qwen3.5-4B BF16 B32/P53/G16, 24 GDN layers and15 incremental decode steps, TP1, ordinary non-speculative decode. C means the already tested combined S+P128 option, not stock B. Existing source/results unchanged.

## Actual execution evidence

Saved matched_runtime_profile_v1 trace has six complete requests. vLLM has2160 packed-decode GDN and2160 convolution-update kernels, establishing360 of each per request. B has2160 decode_vstream,2160 batched convolution-update and2304 separate gate-preprocessing kernels (384 per request, including24 prefill). Corresponding six-request summed kernel durations: vLLM packed323.098ms/conv9.639ms; B recurrent353.241ms/conv15.812ms/gate2.800ms. These are instrumented sums, not standalone or primary wall-time effects; packed includes gate work while B recurrent does not. No equivalent C trace was collected here.

## Processing flow after projection

Dimensions: Hk=16, Hv=32, K=V=128, Q/K each2048, V/Z each4096, mixed QKV8192, A/B each32. Projection inputs/outputs are BF16 in this workload; state math is FP32.

B: separate/grouped projection results -> mixed QKV [32,8192,1] -> batched causal conv+SiLU and rolling conv-cache update -> transpose/split/reshape into Q,K [32,1,16,128], V [32,1,32,128] -> separate gate preprocess -> six existing typed contiguous adapter calls (Q,K,V,g,beta,initial_state) -> decode_vstream -> new FP32 final-state allocation ->64MiB copy into stable cache -> gated RMSNorm(core,z) -> output projection. No physical Q/K head repeat at decode; value head hv maps to key head floor(hv/2).

C: same decode projections/conv/gates/typed-copy sites/arithmetic as B, but S uses the existing cache as one StateInOut operand and skips its final-state copy. The same cached Tensor object and storage pointer remain stable. P128 changes prefill only, not this decode algorithm. Prefill24 state copies remain.

vLLM: merged projection [q,k,v,z] and [b,a] -> split views (A/B made contiguous) -> packed mixed QKV [num_actual_tokens,8192] -> indexed convolution-update+SiLU -> packed GDN reads Q/K/V offsets directly, fuses gate generation and Q/K normalization, reads and updates indexed persistent state -> writes caller-owned output buffer [32,1,32,128] -> gated RMSNorm(core,z) -> output projection. Trace proves packed kernel ran, not that any two engines' intermediate tensors were equal.

After B typed copies Q/K/V are standard contiguous logical tensors; state strides are (524288,16384,128,1). vLLM accepts explicit mixed-QKV/A/B token strides (last dimension contiguous); exact live numeric token strides and cache-slot mapping were not captured. Do not infer them solely from dimensions or assert packed data is row-contiguous. Checkpoint mamba_ssm_dtype=float32 and vLLM's Qwen3.5 config update select FP32 recurrent storage; live pointer/stride/dtype capture is still required for a strict call contract.

## Crucial state-axis distinction

TileIR reads/stores State[b,h,k,v], with offset ((b*Hv+h)*K+k)*V+v. Public decode source ct.load/store index=(idx_b,idx_hv,0,idx_sv), tile=(1,1,BLOCK_K,STREAM_V_TILE), shows V is contiguous.

vLLM packed source computes p_h=base+slot*stride_state_token+h*V*K+v*K+k (fused_recurrent.py approximately303–306 and335–337). Its state is [slot,Hv,V,K], K contiguous. Thus canonical conversion is S_tile[b,h,k,v] = S_vllm[slot(b),h,v,k]. K=V=128 does NOT make the physical layouts equivalent. Cache slot0 is invalid/padding in vLLM and produces zero output; B has direct batch indexing and no corresponding slot sentinel behavior.

The mathematical update in TileIR orientation is H'=gamma H + k outer delta, delta=beta*(v-k^T gamma H). vLLM stores its transpose and uses delta outer k. A raw state comparison without slot gathering and last-two-axis transpose is invalid.

## Numerical differences, not just layout

1. B/C beta=sigmoid(b) is computed FP32 but stored in BF16 beta_out before recurrent reload. vLLM packed keeps sigmoid(b) FP32 in-register. B gate output g is FP32.
2. B softplus is log(1+exp(x)); vLLM uses x directly above threshold20. Overflow/extreme and rounding behavior therefore differ.
3. B Q/K normalization uses reciprocal-sqrt multiplication, vLLM sqrt division, both epsilon1e-6. They need not be bitwise equal.
4. B decode computes output as OutBase + Delta*(k dot q); vLLM computes the dot of the updated state with q. These are algebraically equivalent in real arithmetic but have different FP32 operation/reduction ordering.
5. Conv, sigmoid/exp, SiLU and norm implementations can use different arithmetic contractions/approximations. Equal token IDs are much weaker than intermediate raw equality.

## Convolution/cache ownership

B keeps BF16 conv_state [32,8192,4], reads columns1/2/3 plus current x and shifts to [old1,old2,old3,x]. vLLM ordinary non-spec cache shape uses width-1=3 history values, plus optional layout orientation and cache-slot indirection. Its kernel reads three historical columns and appends current x. For canonical history comparison, gather vLLM slot and compare against B's last3 columns, not all4. Both mutate history in-place; physical shapes/ownership contracts differ.

B returns a newly allocated final-state tensor from recurrent execution, then copies its value into a stable cache. C returns that stable cache itself, so its low-level final-state alias contract differs from B even though the model81 raw outputs/cache pass. vLLM wrapper passes h0=ht=initial_state and returns (out,initial_state), using caller output/cache buffers. C does not implement vLLM block-index mapping, arbitrary cache layout, speculation, prefix reuse or multi-stream scheduler ownership.

## TileIRBLAS boundary and unsupported claims

Input/output projections are logical GEMMs and belong to TileIRBLAS matrix contracts. Conv/history update, gate functions, normalization, recurrent outer-product/reduction and indexed cache management are separate stateful auxiliary APIs. Calling all of these 'same cuBLAS functionality' would conflate contracts.

Existing2304 direct cuBLAS comparisons and model81 raw equality qualify the tested TileIR vs its hybrid native baseline, not vLLM GDN intermediates. The vLLM comparison established identical generated token IDs only. No claim of vLLM intermediate bitwise equivalence is justified.

## Next verification plan, not executed

At one exact layer/decode step, capture the same semantic projected Q/K/V/Z/A/B and pre-state/history before mutation. Record actual dtype/shape/strides/offsets, slot indices, scaling/epsilon and alias relationships. Canonicalize state via slot gather+transpose and convolution history via last3 selection. Compare in order: conv output/history, FP32 beta/g before quantization, normalized q/k, decayed state, delta, final state, output, gated norm, projection. Use raw equality, numerical tolerance and independent FP64 reference as separate verdicts. Feed identical inputs/settings to both public functions, rather than comparing two diverged model trajectories. Include extreme gates, nonfinite policy, distinct slots, invalid-slot behavior and non-square K/V in an auxiliary-contract test where supported. No new GPU experiment is authorized by this report alone.

## Source anchors

- TileGym transformers/qwen3_5/modeling_qwen3_5.py:165–218 for split/preprocess/decode/cache/norm.
- TileGym ops/cutile/recurrent_gated_delta_rule.py:230–357 for exact state indexing and arithmetic.
- TileGym transformers/qwen3_5/kernels/gdr_preprocess.py:29–62 for beta cast and softplus.
- tileir_decode_conv_batch_v1/kernels.py: batch convolution/history implementation.
- vLLM qwen_gdn_linear_attn.py:985–1058 projections/dispatch,1730–1780 packed path,932–944 output norm/projection.
- vLLM third_party/flash_linear_attention/ops/fused_recurrent.py:256–485 packed kernel, layout and API.
- vLLM mamba_utils.py:280–303 state shapes; models/config.py:853 onward Qwen dtype update.
- vLLM mamba/ops/causal_conv1d.py:883 onward history index/rolling,1119 onward API strides.

Exact installed source paths and SHA256 values are preserved in SOURCE-PINS.json.
