# Five-case accumulation grouping assessment: completed, not adopted

**Conclusion:** changing general FP32 grouping changes BF16 rounding, but is not a universal cuBLAS-compatibility fix. Partial4 fixes the newly observed ba channel8 mismatch while introducing raw disagreements on three other input matrices. Partial2 fixes that original channel but introduces a smaller disagreement elsewhere. Neither candidate is adopted; no performance measurement was made.

## Verified execution
One grouping01 worker actual exit0. Five cases complete, all inputs unchanged, reference and each candidate repeat bitwise. All compared outputs finite. Parent cleanup confirms child absent, no residual process group and lock reacquired; independent recovery receipt linked below. Original kernels, captured banks, prior tolerances and HOLD evidence remain unchanged.

## Actual input scope
These are five actual X/W matrices, not five independent populations or every Qwen input. Prefill1696 rows contain repeated requests; ba prefill has43 unique X row bitpatterns, gate_up prefill53. All three decode matrices have32 logical rows but just1 unique row each. Thus32 identical mismatch rows are not32 independent numerical cases.

## Versus direct public F99/math16
Cells are **raw mismatch count / strict tolerance failures / prior BF16 tolerance failures**. Strict remains1e-5+1e-5|F|; prior BF16 remains0.01+0.01|F|. No threshold was loosened.
|Case (M,N,K)|Unique X rows|Original|Partial2|Partial4|
|---|---:|---:|---:|---:|
|ba_prefill (1696, 64, 2560)|43|0 / 0 / 0|64 / 32 / 0|96 / 64 / 0|
|ba_decode (32, 64, 2560)|1|0 / 0 / 0|0 / 0 / 0|0 / 0 / 0|
|qkvz_decode (32, 12288, 2560)|1|0 / 0 / 0|256 / 96 / 0|352 / 160 / 0|
|gate_up_prefill (1696, 18432, 2560)|53|0 / 0 / 0|28384 / 14528 / 0|40192 / 21312 / 0|
|ba_decode_layer1_mismatch (32, 64, 2560)|1|32 / 32 / 0|32 / 0 / 0|0 / 0 / 0|

All prior-BF16 tolerance failures were0, but that does not erase raw or strict failures. Original matched F99 on the four earlier bank matrices, not on the newly observed layer1 ba matrix. For that new matrix: original maxabs0.00048828125 (32 strict failures); partial2 maxabs0.00000095367431640625 (32 raw differences but0 strict failures); partial4 raw matches F99 on all2048 outputs.

## Selected exact oracle (not global accuracy)
Coordinates are a deliberately nonuniform union selected from first differences, largest absolute differences, smallest-reference-magnitude differences as a cancellation proxy, plus a few matching controls. At most8 points/case; duplicate rows may still occur. Exact BF16 dyadic dot→RN-even is evaluated only there. FP64 is separate, not exact-real.
|Case|Selected points|F99 exact-rounded matches|Original|Partial2|Partial4|
|---|---:|---:|---:|---:|---:|
|ba_prefill|6|2|2|5|6|
|ba_decode|2|2|2|2|2|
|qkvz_decode|7|2|2|5|5|
|gate_up_prefill|8|2|2|6|6|
|ba_decode_layer1_mismatch|6|6|4|4|6|

The partial variants often agree more closely with the exact rounded dot on these selected elements while agreeing LESS with cuBLAS over the full matrices. These are different targets. This biased sample cannot establish a winner, aggregate error reduction or universal numerical superiority. The original F99 result is not defined to equal exact mathematical dot rounded once.

## What changed and what did not
Source variants preserve logical NT matmul, BF16 inputs/output, geometry and final RN cast. Eighty K32 tiles are assigned to2/4 round-robin FP32 accumulators; each tile retains two ascending K16 MMA updates. Their final2/4-way FP32 tree adds new rounding points. This demonstrates association sensitivity, not vendor-internal ordering or an inferred proprietary algorithm. No one-channel correction, source replacement, new library contract or model-level adoption occurred.
Public reference is the separately created HOST-pointer handle, explicit current stream, alpha1/beta0, compute32F, algo99 and verified math16. Native handle/workspace physical identity is not asserted. Older capturedY is kept separately from this reference. No time or throughput claim follows from this correctness run.

## Provenance
- RESULT SHA256: 39e99f412575d7de23e2da58fc29d8e676e122cabb1dec9b6d137ed04fabf7df
- Launcher receipt SHA256: f6be0fd1504a1a9844b4c9059fd9d7ee292301cc209f7df2da51f1cf8d8fac4e
- Independent recovery SHA256: 403031b39e4a3ce33fa5fe626f1a34e0c276ad98c02d3576ca70cb22d7b5dfbe
- Source CPU-PINS SHA256: cd0cdfb645286a86377859892cb4ad25cb9dd0a0508cebe370bc560f0462c77e
- All currently pinned source hashes rechecked unchanged during this report. Detailed source/input paths, saved output artifact hashes and per-element oracle evidence remain in original RESULT.
