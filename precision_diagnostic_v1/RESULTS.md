# Public precision-mode diagnostic, actual Qwen inputs

Same captured decode inputs, weights, FP32 compute type and default GEMM algorithm; same C allocation reset to NaN for every call. Public cublasSetMathMode/cublasGetMathMode select DEFAULT=0 versus DISALLOW_REDUCED_PRECISION_REDUCTION=16. Two reference calls per mode confirm repeatability. TileIR source is unchanged. Three cases only; no model or performance result.

|Case|TileIR vs default mismatches|TileIR vs mode16 mismatches|TileIR vs mode16 max abs|default vs FP64-rounded mismatches|mode16 vs FP64-rounded mismatches|TileIR vs FP64-rounded mismatches|
|---|---:|---:|---:|---:|---:|---:|
|ba|768|0|0.0|768|0|0|
|K|11232|64|7.62939453125e-06|11296|96|128|
|V|11648|64|0.000244140625|11648|32|32|

For ba M32/N64/K2560, changing only the public math-mode flag removes all768 differences. TileIR, mode16 and the FP64-rounded BF16 oracle are raw equal. Captured vLLM ba equals default cuBLAS, not mode16. This establishes the numerical difference is controlled by this public precision policy for these inputs; it does not reconstruct proprietary internal code or prove a universal algorithm identity.

For attention K/V M32/N1024/K2560, the flag removes nearly all large discrepancies but64 raw differences remain in each. Residual max abs is7.62939453125e-06(K) and0.000244140625(V). Both are much smaller than0.015625 in default comparison. Floating-point reduction ordering is a possible explanation, not directly proved. Captured runtime outputs were slices of one packed GEMM, not separately executed K/V references, and equal TileIR here. FP64 is an independent high-precision numerical reference, not a formal exact-real certificate.

Both reference repeats are raw equal and all inputs are unchanged. Child and launcher actual exit0, empty owned process group, lock reacquired and separate recovery confirmed. Receipt raw_pass only means this diagnostic completed; it is NOT a new all-input numerical equality certification. The prior eight-case raw-HOLD result is unchanged.

Next runtime comparison must separate native default precision from native FP32-reduction and TileIR FP32-reduction contracts. Do not silently relabel the unchanged TileIR implementation as default-cuBLAS bitwise compatible, or compare timing with precision differences hidden.

Public specification: NVIDIA cuBLAS13.1 documentation, GEMM Algorithms Numerical Behavior / cublasMath_t / cublasSetMathMode:
https://docs.nvidia.com/cuda/archive/13.1.1/cublas/index.html#gemm-algorithms-numerical-behavior
The public flag requires reductions to use the accumulator type rather than the narrower output type. Exact installed enum values and setter/getter declarations are pinned from the installed public header. No private ABI, SASS or proprietary binary analysis was used.

RESULT SHA c6be51a340802e39ae7500af7bbc6b75eb42466655f50bd750e6af4082f089cd; independent recovery SHA7e2adb1b3c3c3e141204efee5d7ea2ba2ba8b45e2a481b437e4dc3e19e14fa9f. Scope remains the three captured cases from one repeated-prompt workload.
