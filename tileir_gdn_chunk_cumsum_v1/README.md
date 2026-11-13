# Experimental first GDN prefill stage: chunk-local scalar cumsum

CPU-prepared only. GPU compile, numerical replay, runtime integration and performance are NOT performed. Actual selected backend and metadata are pending parent's native capture. No fallback and no automatic runtime patch.

Public source provenance: installed vLLM 0.32.0 `third_party/flash_linear_attention/ops/{chunk,cumsum}.py`, pinned alongside the remaining stage sources. Apache-2.0 vLLM notices; copied FLA source original MIT copyright (c) 2023-2025 Songlin Yang, Yu Zhang. This implementation is independently expressed in cuTile; license text is included. No proprietary code was used.

Six stages in source order: (1) g inclusive chunk-local cumsum64 FP32, (2) KKT scaled dot FP32, (3) unit-lower triangular solve with BF16 output, (4) WY w/u recomputation, (5) h/v_new/final-state recurrence, (6) o projection. Optional QK L2 normalization is disabled by the model's already-normalized caller. State gather/scatter is outside this chunk boundary. Indexed decode is a different fused function and is not implemented here.

Candidate API fixes expected g FP32 [1,1696,32], contiguous offset0, 32 sequences length53, chunk64, forward/non-head-first. Input g is read-only; output is fresh matching-shape FP32. cu_seqlens and chunk_indices are checked for exact expected values; this host validation synchronizes and is not yet a fast-path design. Expected metadata is source-derived, NOT an observed descriptor until capture confirms it.

cuTile cumsum RN implements the same mathematical operation but scan FP32 addition ordering may differ from native Triton. No bitwise equality is claimed before direct native same-input replay. On mismatch preserve result, inspect only public generated Triton layout/IR and express corresponding ordered scan if justified; do not relax raw/strict/finite gates. Masked padding beyond each sequence end is zero and never stored.

Next qualification: verify capture selected forward_native/triton, descriptors and source pins; load saved g/cu_seqlens/chunk_indices; native scalar cumsum twice then candidate twice with input-preservation/fresh-disjoint-output/shape/dtype/stride and raw/strict/finite/repeat gates. Stop on any failure; later full24 shadow before replacement. Remaining five stages must be separately implemented and qualified, not represented as complete by this candidate.
