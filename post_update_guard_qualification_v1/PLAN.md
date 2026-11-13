# Public staged post-update value validation

CPU-prepared, GPU/compiler/model unqualified. No existing source is modified.
Public source basis: `tileir_vllm_post_update_v3/contract.py` values and metadata,
`residual_four_gdn_owned_lean_v2/four.py` owned caller, public cuda.tile gather,
comparison, integer cast and reduction. No proprietary binary evidence is used.
The original numerical post_update kernel and owner/UVA lifetime policy are unchanged.

## Fixed contract and order
Exactly the existing 10 disjoint full-backing tensors from OBSERVED.json are admitted.
32 rows, idx/sampled int64; computed/hist/ns/nr/q/total int32. Every arithmetic input
is promoted to int64 before addition/subtraction. Stage errors, in exact order:
1 state index bounds; 2 conflicting active state indices; 3 sample count exceeds
observed row; 4 append bounds; 5 histogram/token bounds; 6 histogram integer overflow;
7 query/rejection arithmetic overflow (query only); 8 same message (delta only);
9 computed integer overflow.

Each stage launches separately and reads its int32 status as a host integer before
any next stage is enqueued. Thus stage9 is never entered after stage7 failure,
and dependent histogram reads never precede valid state/count/token checks.
Negative state IDs remain inactive, including INT64_MIN. Duplicate inactive IDs,
invalid inactive token values and arbitrary inactive arithmetic do not fail validation.
Negative histogram values are permitted exactly as in the original guard.

This is not an arbitrary masked range API and does not duplicate the independently
prepared int64 all_in_range primitive. Each source stage reduces a fixed 32-vector;
uniqueness uses a bounded 32x32 equality matrix. Inactive entries may perform safe
loads from existing element0 of total/computed/hist; their values never affect status.
This is extra bounded readonly access, not address reconstruction or a payload copy.
No invalid index is dereferenced. Inputs and independent status allocation are held
strongly through status synchronization. Uncertain enqueue failures permanently latch
the Guard and retain pending owners. No fallback or retry is provided.

## Owned caller bridge
The exact current Replacement.call source AST is copied, and exactly one
`validate_values(torch,args)` expression is changed to an injected callback.
Original metadata checks, error wrapper, post-update arithmetic call, row census,
owner retention and storage/return checks remain in the copied body. This instance
method override is not a global module/torch patch. Existing instance overrides are
rejected. Reverse close removes only the installed owned method; tampering is an error.
A CPU mock executes the actual public caller to the callback boundary and checks
that numerical work is never reached after the test callback raises.

## Isolated qualification scope
`post_update_guard_qualification_v1` prepares 26 private fixtures with fixed metadata.
Native2 and candidate2 use exactly the same allocations, complete input backing bytes
are checked after every invocation, and the current stream must remain unchanged.
Success and first ValueError messages must agree. Failed candidates must report an
exact stage prefix and must have no pending owner after a completed false predicate.
Cases include duplicate/state/count/append/token bounds, INT32 and INT64 extrema,
histogram negative/max, query/delta/computed overflows, first-failure precedence,
all-inactive and masked inactive extreme values. Hist backing is 31,784,960 bytes;
fixtures are sequential, not simultaneously resident. The whole backing snapshots
are diagnostic and are not part of production-like performance.

Existing isolated launcher lifecycle is reused: common lock, exact GUI admission,
fresh local cache folders, worker585s/parent600s, process cleanup and source pins.
No GPU run is authorized by CPU readiness alone; coordinator owns actual execution.
Model16/original UVA addresses, combined helper compatibility, new residual profile,
and performance are all pending. Prior 1,216 native kernels are old-profile evidence,
not a measured savings estimate. Nine source launches per valid guard are not a
physical kernel-count measurement. No 100% coverage claim.

## CPU evidence
14 standard-library tests passed, including all 26 oracle cases, every fail-fast stage,
exact public caller AST/restore, invalid status, fail-before-dependent-hist access,
and strict qualification result-gate negatives. No torch/CUDA imports or compilation.
Coordinator released the GPU window for CPU freeze. Final CPU receipt records
pins/manifest/two dry entries; actual GPU and model qualification remain pending.
