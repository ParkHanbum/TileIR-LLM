# First native-trajectory projection mismatch: linked result

- Native F diagnostic and isolated replay both completed with actual child exit0, adapter/precision restoration and independent process/lock recovery. New work here was CPU-only exact-oracle analysis (exit0), no GPU rerun.
- First 180 selected calls: **1,674,414,080 BF16 outputs raw-exact**. Call181 (zero-based180), `language_model.model.layers.1.linear_attn.in_proj_ba`, M32 N64 K2560, is the first selected mismatch. This is second decode invocation by 88 selected modules per model step (88 prefill +88 first decode +5th selected projection next step).
- **32/2048 differences**, all row0..31/column8; finite, max absolute error0.00048828125 (one adjacent BF16 step here). Bound native repeat is raw-exact to original native. Remaining1227 selected calls return native but are not shadowed.
- Native request completes1408 selected calls and512 tokens. Its512 tokens and2560 public top5/chosen-token logprob scalars match prior F request exactly. This checks recorded output equivalence under instrumentation, not universal absence of side effects.
- Isolated public cublasGemmEx algo99/math16 twice matches saved native; TileIR twice matches saved TileIR. Inputs unchanged in isolated replay. New handle/default workspace are not original vLLM handle/workspace; actual native repeat is the direct same-bound-call evidence. Logger covers repeated native call only.

## Expanded exact oracle
Existing CPU `exact_dot` was run on **all32 differing coordinates**, with saved input hashes validated. Exact finite BF16 dyadic multiplication/sum, then BF16 round-nearest-even, matches native/nativeRepeat32/32 and TileIR0/32. These32 coordinates have 1 unique exact sum, so they are not32 independent numerical cases.
Example exact dot ≈-0.11059498259419343; native=-0.1103515625, TileIR=-0.11083984375. Their BF16 midpoint is -0.110595703125; exact-minus-midpoint is 7.2053080657497048e-07. Exact lies on native's side.

## What this establishes, and does not
The logical public contract remains the same bias-free BF16 NT linear operation with FP32 compute; finite-precision accumulation/reduction ordering may differ. Results establish a reproducible one-BF16-step output disagreement on the same native-trajectory operands. They **do not identify** which intermediate accumulator or conversion caused it. Public operation order and final BF16 rounding must be separated; forthcoming FP32-output probe may distinguish them. Do not label final conversion itself faulty before that probe.
This is only the first mismatch among selected packed projections on **native F trajectory**. It does not locate the first difference on T trajectory, prove this boundary alone caused final logprob differences, characterize nonselected operators, establish whole-model accuracy, or demonstrate cuBLAS globally more accurate. No timing result is inferred from diagnostic request duration. Existing default-mode qualification raw HOLD and F/T logprob HOLD remain unchanged.

`LINKED-RESULT.json` links counts and comparisons. `cpuExact32/RESULT.json` preserves all32 exact sums and output bits. `SOURCE-HASHES.json` and `SEAL.json` bind original evidence and new analysis.
