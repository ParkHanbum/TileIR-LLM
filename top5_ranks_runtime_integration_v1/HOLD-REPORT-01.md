# Sampler three-suboperation integration: HOLD

Actual `samplerThreeIntegration01` exited 1 during native-shadow; complete=false. No replacement request ran. All 991 frozen source pins remain unchanged, and independent recovery confirms no child/process group, free lock and unchanged original cache.

The first six helper calls passed. Call 7 (decode, num_computed_tokens=58 for all 32 identical requests) failed top5 token-ID raw equality and the explicit unique-top5 gate. Selected top5 positions 3 and 4 each have two equal-valued candidates in every row. Ranks remained raw-exact and selected FP32 score arrays remained raw-exact (strict failures 0, max absolute difference 0, finite). Equal score arrays do not make differing token IDs semantically interchangeable.

Observed counts are 7 native helper calls plus 7 candidate helper calls in shadow: 21 named logical suboperation invocations per side (top5, ranks, selected logprobs). This is neither 16-call qualification nor full-model public-output equivalence. Replacement count is zero. No performance, ABI, optimized-fusion or general tie-order compatibility claim. The earlier first/last-bank isolated successes remain valid only for those banks.

The original raw/tie gates and all numerical APIs are unchanged. Separate ranks+logprob v2 retains native topk and remains a distinct experiment; this report does not promote its unexecuted status to success. Top5 tie-order investigation is assigned separately.

Evidence: RESULT.json, shadow/shadow-PROGRESS.json, first_mismatch/values.pt, launcher and independent recovery receipts. This append-only report performs no GPU work or experiment retry.
