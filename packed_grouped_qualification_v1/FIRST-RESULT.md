# First actual-vLLM packed/grouped qualification

Status: all eight contract and predeclared diagnostic numerical checks passed. Raw-equality gate is HOLD, actual child/launcher exit2 by design, not a crash. Timing/model performance has NOT run for this extension. Original results remain unchanged.

The actual inputs are four packed projection families × prefill1696/decode32, K2560, BF16; recorded X stride[2560,1], Y stride[N,1]. Eight packed outputs contain70,889,472 elements. Direct reference is public cublasGemmEx BF16/COMPUTE32F, logical NT, alpha1/beta0, DEFAULT_MATH and default algorithm. Input/weight preservation, output segment stride/alias, supplied contiguous grouped output and negative padded/shared-output rejection all pass. All public reference repeat checks pass raw equality.

|family|phase|packed vs direct raw|grouped vs segmented direct raw|packed/reference vs captured runtime raw|packed vs captured mismatches|max abs|
|---|---|---|---|---|---:|---:|
|qkvz|prefill|True|True|True|0|0.0|
|ba|prefill|True|True|True|0|0.0|
|gate_up|prefill|True|True|True|0|0.0|
|attentionqkv|prefill|True|True|True|0|0.0|
|qkvz|decode|True|True|True|0|0.0|
|ba|decode|False|False|False|768|0.015625|
|gate_up|decode|True|True|True|0|0.0|
|attentionqkv|decode|True|False|True|0|0.0|

Decode ba:768/2048 packed outputs differ from both captured vLLM and direct full cuBLAS, maximum absolute difference0.015625. Full cuBLAS itself matches captured vLLM exactly. Two grouped ba members also differ from their segmented cuBLAS references.
Decode attentionqkv: packed output matches full cuBLAS and captured vLLM exactly, but grouped K/V outputs differ from segmented cuBLAS by11232/11648 elements, max abs0.015625 each. TileIR grouped K/V match their packed TileIR views exactly. Thus packed versus segmented vendor dispatch is itself a distinct numerical comparison.

All eight satisfy the previously declared diagnostic abs_error <=0.01+0.01*abs(reference), finite/no-nonfinite condition. This threshold is NOT a universal accuracy guarantee, and raw differences are not hidden by tolerance. The next step is same-input public cuBLAS math-mode controls and independent FP64 reference, not proprietary implementation analysis or premature model timing.

Group coverage is the captured packed segments. Original B GDN(qkv,z,b,a) tuple and KV-only grouped submission are not directly covered as those exact grouped API calls. ABI compatibility is not provided: these are Python adapter functional APIs using TileIR kernels.

Artifacts: actual input manifest SHA719a7fb82ab3f9be0a9ee8383226a47811bf5ba8686542347a0c3b6e35bd6a08; RESULT SHA d3121922f7ac6396d42bcaa621d5a64f29550142dceb7bf5a258d92c722adac9; independent recovery SHA69e569035a3792ec28353c9ae86abab22633c695b8b62fdc7ab010ca9b99c64b. Child absent, process group empty, lock reacquired; known GUI retained. No timing claim.
