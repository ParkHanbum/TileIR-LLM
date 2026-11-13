# FP32-output probe: separate linked finding

The probe completed with actual child exit0 and independent process/lock recovery. All original analysis-sealed files and probe source pins were verified unchanged. No new GPU experiment was run for this report.

## Verified result
For all2048 output elements, FP32-output probe followed by explicit cuTile BF16 RN conversion is raw-identical to the original TileIR result and the saved TileIR result. Inputs were unchanged; FP32 outputs are finite.
All32 previous mismatch coordinates (rows0..31,column8) were checked against the separately saved exact integer oracle. They share one numerical case:

- Probe FP32 accumulator: **-0.11059573292732239**.
- Exact real dot of captured BF16 operands: **-0.11059498259419343**.
- Adjacent BF16 midpoint: **-0.110595703125**.
- Accumulator minus exact: **-7.503331289626658e-7**.
- Accumulator minus midpoint: **-2.9802322387695312e-8**.
- Exact minus midpoint: **+7.205308065749705e-7**.

Thus this probe's small accumulation error moves the value to the other side of a BF16 rounding boundary. Its explicit RN conversion consistently selects the saved TileIR value -0.11083984375, whereas exact-dot RN selects the native value -0.1103515625. The evidence supports accumulation-side sensitivity, not a faulty final RN conversion in this probe.

## Interpretation limits
The public source accumulator initialization, K loop, operands and two `ct.mma` updates per tile match the original source. The probe stores FP32 rather than BF16, which can change compiler lowering. It is **not direct measurement of the original kernel's internal accumulator**, nor proof of identical compiled accumulation order. Public source order does not expose implementation order inside MMA or inside vendor kernels. We did not reverse engineer vendor internals.

The same output at2048 elements and the exact32-coordinate arithmetic constrain the explanation but do not identify the precise original MMA/reduction ordering. These32 coordinates repeat one case, not32 independent test inputs. This does not establish global model quality, the entire final logprob mismatch's causal chain, or universal superiority of either implementation.

No library change was adopted. No timing or performance claim is made. Existing default-mode raw HOLD and F/T runtime logprob HOLD remain unchanged. This report adds evidence without replacing earlier reports or seals.
