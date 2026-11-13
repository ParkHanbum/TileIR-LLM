# K128 BF16 partials: one-contract hypothesis

This is one new deterministic general arithmetic recipe, not a fitted per-channel correction. Public API dispatch is restricted to observed BF16 NT (M,N,K)=(32,64,2560). All other shapes call the unchanged existing public TileIRBLAS implementation; this is not vendor fallback. Default True/True and native D99/math0/compute68 reference are not modified.

For each contiguous K128 block, accumulate eight K16 MMA steps in FP32 (four K32 loads), round that partial to BF16 RN, convert to FP32, then add it to the ordered FP32 sum of20 blocks. Round the final sum to BF16 RN. BM32/BN128 retains the earlier blocked geometry with masked output columns for N64. No data-dependent branch, channel coefficient or golden-output use exists in the candidate.

Selection evidence: 1 unique input vector repeated over32 batch rows,64 channels. One of24 CPU exact-partial rounding/reduction recipes matched the captured default native outputs in all64 channels. This is exploratory selection bias, not an independent test or model proof. CPU exact partials are NOT the actual MMA partials; hardware MMA accumulation can differ, so CPU match does not imply GPU success. Earlier contiguous half-K split2 variants failed and are not rerun.

Validation order: selected ba_decode first, separate layer1 ba_decode as the only new-arithmetic holdout, then ba_prefill/qkvz_decode/gate_up_prefill as unchanged-route regression controls. The layer1 captured native Y came from historical F mode; it is not the adoption reference. Fresh public D99/math0 native repeat on those same saved operands is the reference. Other selected/regression native calls must also reproduce their stored default outputs. Existing unchanged TileIR outputs must reproduce saved baselines. First failed candidate stops the single candidate; no relaxed raw/strict gate. No GPU/compile has occurred in preparation.

A sampled pass would not establish all360 ba decode calls or all-input bitwise equivalence and would need later default-mode live shadow before replacement. Performance is not tested here.

Qualifier reuses the prior five-matrix control and lifecycle harness. Old split2 sources/results remain untouched. At most2 native+2 unchanged baseline+2 candidate calls per case,30 logical GEMMs; worker570s/parent600s, compile90s. Exact capture layout/backing hashes and default flags remain checked. Baseline reproduction is separate from new candidate comparison. No model/timing.
