# Nine-adapter CPU structure, not released

Native default True/True read-only policy, same eager B32/P53/G16 workload. Reuse seven qualified implementations plus QK128 and SiLU512 by reference. Total2256 logical calls, not GPU events or time coverage. No numerical API edits. No runner or GPU release exists in this draft. Final actual component proofs, source pins, all-subgate result auditing and independent review must precede any launch. New full2256 same-run shadow is mandatory before conditional replacement.

## Binding interaction
Attention gate installs instance.forward, whose copied public ordering calls `this._project_qkv_gate` by dynamic attribute lookup. It does not wrap the project helper. QK installed later wraps class._project_qkv_gate and the qwen3_next module imported fused helper; its original source checks therefore still see the original project method. The attention forward calls through the new QK wrapper. QK returns the same project tuple entries and validates V packed-view ownership. SiLU wraps MLP act_fn instance.op only and has no overlap. Reverse order closes SiLU, QK helper/project, samplecounts, gated, positions, slots, gather, attention instance.forward, norm. Thus source-level binding domains are compatible; actual numerical composition is NOT yet verified.

## Cleanup
Use frozen public_vllm_composed_integration_v5.cleanup.final_cleanup directly (SHA7be9d861bc92eb138ff4424805637eee97f0e83ed5586002f0b5111e41c0985f). Read-only precision drift is HOLD without setter repair and does not skip later cleanup attempts.
