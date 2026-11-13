# Actual public cuBLAS GemmEx contract: D vs F

Both official logs contain2448 GemmEx calls; nonpointer public-contract sequences identical: **True**. Contracts compared: transa, transb, m, n, k, Atype, Btype, Ctype, lda, ldb, ldc, computeType, algo.

## Recorded policy at every GEMM
- D: 2448/2448 CUBLAS_DEFAULT_MATH.
- F: 2448/2448 CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION.
- Both: A/B/C CUDA_R_16BF(14), computeType CUBLAS_COMPUTE_32F(68), transa T/transb N, algo CUBLAS_GEMM_DEFAULT_TENSOR_OP(99). These are checked from all call records, not inferred from a setter.

## Counts by actual column-major API dimensions
|m|n|k|lda|ldb|ldc|count per arm|
|---:|---:|---:|---:|---:|---:|---:|
|12288|1696|2560|2560|2560|12288|24|
|64|1696|2560|2560|2560|64|24|
|2560|1696|4096|4096|4096|2560|32|
|18432|1696|2560|2560|2560|18432|32|
|2560|1696|9216|9216|9216|2560|32|
|10240|1696|2560|2560|2560|10240|8|
|248320|32|2560|2560|2560|248320|16|
|12288|32|2560|2560|2560|12288|360|
|64|32|2560|2560|2560|64|360|
|2560|32|4096|4096|4096|2560|480|
|18432|32|2560|2560|2560|18432|480|
|2560|32|9216|9216|9216|2560|480|
|10240|32|2560|2560|2560|10240|120|

## Address/context comparison
Pointer-string differences by call: {'handle': 0, 'alpha': 0, 'A': 0, 'B': 281, 'beta': 0, 'C': 475}.
Context differences: {'process': 0, 'thread': 0, 'gpu': 0, 'handle': 0, 'stream': 0, 'MathMode': 2448}.
Handle/stream groups D: [{'handle': 'POINTER (IN HEX:0x0x56047ec0)', 'stream': 'POINTER (IN HEX:0x(nil)) (defaultStream)', 'count': 2448}]; F: [{'handle': 'POINTER (IN HEX:0x0x56047ec0)', 'stream': 'POINTER (IN HEX:0x(nil)) (defaultStream)', 'count': 2448}].
Alpha/beta entries are pointer-only, so even equal addresses do not establish equal scalar contents. No numeric scalar is inferred. The logger reports defaultStream; the original raw representation is retained.

## Interpretation limits
- Alpha/beta values are unknown: only pointer strings logged. Pointer mode not logged.
- MathMode is taken from each GemmEx context, not setters.
- DISALLOW_REDUCED_PRECISION_REDUCTION is a policy restriction; actual intermediate dtype, split-K decision or selected kernel not established by this log.
- CUBLAS_GEMM_DEFAULT_TENSOR_OP(99) is public request enum, not selected internal algorithm ID.
- API m/n refer column-major operands, not logical row-major X/W labels.
- No epilogue field in GemmEx; a no-bias/no-activation model source contract is distinct from an observed runtime epilogue argument.

The data supports policy/contract comparison, not reverse engineering of vendor internals. Source logs are untouched; all calls retain original line number, argument type/raw string, context and block hash. Parser is Python standard library only; no CUDA imports or GPU commands.
