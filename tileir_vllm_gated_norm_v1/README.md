# CPU draft: gated RMSNorm
Only BF16 X/Z[1696 or32,32,128], raw W128 BF16/FP32, epsilon positive finite, silu, norm-before-gate, biasNone, no residual. Lastdimstride1, nonoverlapping positive padded leading strides and storage offset supported by indexed gathers. FreshBF16 output, no input writes. This is not Gemma+1.
FP32 conversions precede reduction/weight/gate; no BF16 intermediate round requested before final output. ct.rsqrt/exp/sum/compiler FMA may differ from Torch native/guarded Triton approximations. Equation parity does not imply raw parity. No compiled code examined, no GPU success claimed.
Actual source contracts and licenses: vLLM layernorm.py Apache2.0; public guarded FLA file Apache2.0 with original MIT attribution toTriDao. This kernel is an independent expression, not copied Triton or vendor binary. Current source/qualification correction: gdn_gated_norm_contract_v1/CURRENT-CONTRACT.md.
Runtime capture source is gated_norm_runtime_capture_v1, twoactuallayer0 calls; metadata+logicalhash/aliases saved. No arbitrary stride guarantee untilactualcapturequalification. CPU7APItests+2observertestsPASS; GPU/compile/capture0.

Tensor subclasses including nn.Parameter are accepted if strided, same-device, non-grad and supported metadata; grad-enabled weight is explicitly rejected. Actual captured weight properties determine runtime scope.
