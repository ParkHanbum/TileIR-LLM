# post_update v3: explicit integer-store cast correction

CPU-only narrow correction for the observed v2 type-check failure. No new compilation/GPU/query/cache work. Existing v2 and four-shadow failure evidence remain unchanged. v3 preserves the same actual-metadata scope, integer-domain guards, original-exactly-once clone-target shadow and cleanup behavior. It does not prove live UVA execution or expand supported helper coverage.

## Sole numerical-source change

kernels.py:26 now stores `ct.astype(token, ct.int32)` into the int32 token-state array. The sampled token stays int64 for last-token storage and histogram indexing. All index arithmetic and every other kernel expression remain unchanged. AST regression removes this one cast and confirms exact equality with v2. contract.py/shadow.py/__init__.py are byte-identical to v2.

## Public native conversion contract

Pinned installed vLLM input_batch.py loads sampled int64 tokens and passes them to tl.store for the int32 all_token_ids backing. Public Triton language/semantic.py `_store_legacy`:1283-1287 casts values to the destination element type before emitting a store; integer conversions at:864-873 use the ordinary integer cast. A narrowing integer conversion truncates excess representation bits rather than saturating. No dependence on out-of-range truncation is introduced here: unchanged qualification guards require every accepted token in [0,248320), entirely within signed int32, so conversion is exactly value-preserving. Negative/too-large sampled IDs remain rejected before enqueue; this is not a general arbitrary-int64 store API. Public cuTile `_stub.py:2648-2672` exposes ct.astype as explicit dtype conversion. No proprietary disassembly or compiler hook used.

## All scatter dtype audit

| Target | Destination | Stored value in v3 | Index |
| --- | --- | --- | --- |
| last | int64 | sampled token int64 | int64 state |
| total | int32 | old_total int32 + n int32 | int64 state |
| tokens | int32 | explicit astype(token,int32) | int64 offset |
| counts | int32 | count int32 + fitting scalar1 | int64 offset |
| computed | int32 | computed int32 + delta int32 | int64 state |

Query endpoints/rejection/delta are int32 in the kernel; existing guard checks widened query-difference, delta and computed sum fit int32. Histogram increment overflow is rejected; total append bounds cap results at128. The int64 row/token index path is distinct from the stored value type and is deliberately not narrowed. The AST fixture models only this specific kernel's input types and public fitting-literal promotion; it is NOT an alternative compiler or proof that no other compile issue exists.

## Verification and next gate

Eleven small stdlib CPU tests pass, including v2 exact mismatch detection, all five v3 stores type-compatible in the fixture, one-line AST equivalence, admitted token endpoints, rejected -1/248320/int32-overflow/int64 extremes, unchanged guards, metadata rejection, original exactly once and fail-stop. Receipts are CPU-RESULT.json/CPU-TEST.log. No torch/CUDA imports occur in tests. Full external artifact/source-tree revalidation is deferred to the coordinator's normal reviewed launcher; the small original-v2 package pins are verified now.

Four-helper author may reuse the existing runner in a fresh version changing only its post package import and pins to tileir_vllm_post_update_v3. No retry happened here. The same limit/census/whole-model correctness/cleanup conditions apply. Compiler acceptance, raw shadow, model outputs and live-UVA safety remain pending until separate authorized execution; earlier native1/candidate-enqueues0 must not count as a candidate pass.
