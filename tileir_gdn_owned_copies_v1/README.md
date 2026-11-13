# GDN owned zero + two strided bit-copy candidates v1

CPU source/AST/mock preparation only. No framework import, GPU query, compilation, cache access or execution. Five stdlib tests PASS. Earlier AST test caught a second zero call in fused-spec-decode; transform now selects ONLY the top-level main-branch assignment, leaving fused-spec-decode untouched. Existing source unchanged.

## Authority and scope

Installed public qwen_gdn_linear_attn.py QwenGatedDeltaNetAttention.forward_cuda lines 1040–1052 explicitly zeros core output and then passes b.contiguous(), a.contiguous() in that evaluation order. Native source comment prohibits replacing zero semantics by uninitialized memory. Candidate uses fresh empty allocation PLUS a zero-fill TileIR launch, not omission of initialization. Separate b then a launches remain in place; no fusion or reorder. The core custom-op reference and binding stay unchanged.

Actual prior metadata: b/a BF16[M,32], stride[64,1], offsets0/32; M1696 or32; core BF16[M,32,128]. Historical lean25 SOURCE-ROLES count: 384 zero +384 b contiguous +384 a contiguous =1152 native kernels, including24 prefill/360 decode. These are old trace observations, NOT a new profile denominator or predicted coverage gain. Runtime qualification and current-call census remain pending.

## Contracts

`OwnedCopies.zeros(shape,dtype,device)` supports only the stated BF16 CUDA shapes, allocates fresh dense output then uint16 zero fills every element. Positive floating zero is all-zero BF16 bits. Neither tensor result nor shape is reused between calls.

`contiguous_b(x)` and `contiguous_a(x)` reject unsupported metadata before allocation/enqueue; accept exact shape/stride/offset, read-only input, no autograd/conj/negative view bits. Fresh output full backing must not overlap input backing. dtype reinterpretation views are same-itemsize uint16, not float conversion. TileIR load/store on uint16 preserves all BF16 raw bits, including NaN payloads and signed zero, subject to future actual compiler/raw-output verification. b/a may share backing with each other as native does: neither is written. Strided ct.load uses supplied 2D tensor descriptor, preserving offset32 of a; no flatten/contiguous operation on input. Source extent is checked. 32-row tiles exactly cover both M values; zero 1024-element tiles exactly cover output.

Native `.contiguous()` can return identity on already-dense input; this narrow adapter rejects such input rather than silently promising generic equivalence. Copy outputs are necessarily fresh for the qualified noncontiguous shape. Kernels use normal bounds behavior; no unchecked access optimization. API stores launch attempts/enqueues and latches any uncertain launch failure; never retries. Unsupported metadata aborts this qualification rather than falling back invisibly. Allocation failure has no enqueue. External controller must fail-stop on any API exception, including pre-enqueue validation.

## Localized composition

`transform_forward(ast_node)` accepts a deepcopy of the already owned forward definition. It replaces exactly one top-level main zero assignment and exactly two core-call argument expressions. It does not install a hook or mutate torch, class methods, existing closures, global bindings or current GDN-core adapter. Integrator should compose this transform into its existing owned-copy builder, then inject `_owned_gdn_copies` into that private function's namespace. Do NOT install a competing forward wrapper or reconstruct an unpatched original over existing owned changes. If existing owned builder changed the core expression, transform rejects instead of guessing. Verify SOURCE-PINS before composing and preserve existing source-scope entry/exit hooks. Model execution/fallback/cleanup remains coordinator-owned.

## Minimal next review/run

Review uint16 tensor-view support with installed cuTile and current forward builder before GPU. First bounded raw shadow on M32 then1696: independently check zeros raw bits, b/a raw equality to native contiguous, source full-backing immutability, fresh output alias, explicit NaN/signed-zero payload cases. Original native once, candidate independent fresh outputs, no injection into live core until raw pass. Then assess actual current forward census separately. No speed or compile success claimed. CPU tests cover descriptor negatives, full uint16 packing roundtrip and selected strided slices, tile arithmetic, AST main-branch targeting/order, and uncertain-launch latch; they do not emulate GPU lowering.
