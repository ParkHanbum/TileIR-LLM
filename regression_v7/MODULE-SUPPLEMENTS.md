# Independent regression supplements

Baseline tag and all original numeric code remain unchanged. These source-only
supplements require a new committed-HEAD CPU and explicit GPU run; CPU review is
not a GPU PASS. The previous tested milestone is preserved externally under
`v7-regression-artifacts/milestone-6e67c8e-01/`.

| New suite | Narrow standalone scope | Not claimed |
|---|---|---|
| norm_conv | two actual prefill norm boundaries; one actual decode conv; native/v7 six records | gated/decode norm, prefill/post-conv, original unrelated shared owner |
| attention_kv | six actual attention banks x native2/tile2; two actual KV phases x native2/tile2 | full live KV backing; reconstruction uses sentinel/compact storage |
| gdn_prefill | one actual layer0 core, native2/tile2, nine stages/output/state; three private parent fixtures x four | all 24 layers or original 5.4GB shared backing; parent values are synthetic |
| sampling_runtime | actual greedy/Top5/logprob/rank calls 0/15; two derived embedding tables; six synthetic metadata cases | tie call6/call8, RNG, full checkpoint or every metadata contract |

The active original solver is solve_tril_v6, not the older standalone v5.
Embedding reconstructs a 1,271,398,400-byte table from selected checkpoint rows;
full-table input-preservation copies alone entail at least 7,628,390,400 D2H
bytes. These are deliberately diagnostic costs, not performance measurements.

All new suites reuse the same exact-HEAD guards, hash-bound external fixtures,
common lock, fresh cache, T/T precision, failure-stop/sync and independent
recovery. Numeric source is not edited. Corruption checks feed actual CPU
output/state copies through original comparators, not just XOR inequality.
Attention helper extraction is checked against original AST and helper execution.
Receipt gates require exact case identity/order and explicitly recorded metric
contracts. New module tests are added to --cpu; --suite modules runs all seven
suites sequentially. Individual suite names are also supported.

A passing endpoint does not substitute for standalone raw coverage. Remaining
QK/RoPE, SiLU, gated norm, prefill/post-conv, and additional runner APIs still
need separate standalone cases beyond whole-workload evidence. Runtime failure
supervision and final device sync belong to the common runner; not every new
module writes per-stage partial output on failure. Do not claim exhaustive
58-kernel or all-shape independent coverage.

Two preserved historical CPU scripts use unqualified `api`/`run` imports. The
loader binds only their exact HEAD-checked canonical module temporarily while
loading that named suite, then removes the alias. A pre-existing foreign alias
is rejected; the original test sources remain unchanged. Actual tests and alias
cleanup/foreign-alias rejection are covered by a new CPU regression. The first
failed supplement CPU run is retained; no GPU run preceded this fix.
