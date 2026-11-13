# All observed vLLM GEMM contracts: same-run raw-exact replacement PASS

153 selected Linear/LM-head owners × 16 calls = **2,448 shadow and 2,448 actual replacement calls**, across all 13 observed contracts. Shadow compares **2,590,441,472 BF16 output elements**, with zero raw mismatches, strict failures or nonfinite outputs. Replacement performs no selected native GEMM and no selected vendor fallback, and every call’s X/Y hashes match the same-run native shadow. K128 ba-decode is used for 360 calls; the existing public TileIR backend handles 2,088.

Both requests preserve the frozen 512 token IDs and 2,560 observed public logprob values exactly. Default True/True precision flags, restore and shutdown pass. Worker PID 578105 and launcher actually exited 0; independently saved recovery confirms child/group absent and lock available. This audit rehashed all 214 pins and all 1,033 original cache files with zero differences. Fresh preseed receipt targets this exact run folder; no earlier stale receipt is reused.

## Contract coverage (logical M,N,K; calls per request)

| M | N | K | Calls | Backend |
|---:|---:|---:|---:|---|
| 32 | 64 | 2560 | 360 | K128 |
| 32 | 2560 | 4096 | 480 | unchanged_tileirblas_once_occ2 |
| 32 | 2560 | 9216 | 480 | unchanged_tileirblas_once_occ2 |
| 32 | 10240 | 2560 | 120 | unchanged_tileirblas_once_occ2 |
| 32 | 12288 | 2560 | 360 | unchanged_tileirblas_once_occ2 |
| 32 | 18432 | 2560 | 480 | unchanged_tileirblas_once_occ2 |
| 32 | 248320 | 2560 | 16 | unchanged_tileirblas_once_occ2 |
| 1696 | 64 | 2560 | 24 | unchanged_tileirblas_once_occ2 |
| 1696 | 2560 | 4096 | 32 | unchanged_tileirblas_once_occ2 |
| 1696 | 2560 | 9216 | 32 | unchanged_tileirblas_once_occ2 |
| 1696 | 10240 | 2560 | 8 | unchanged_tileirblas_once_occ2 |
| 1696 | 12288 | 2560 | 24 | unchanged_tileirblas_once_occ2 |
| 1696 | 18432 | 2560 | 32 | unchanged_tileirblas_once_occ2 |

## Scope and limits

- One eager Qwen3.5-4B BF16 TP1 workload, B32/P53/G16, 32 identical prompts; not diverse inputs or arbitrary dimensions/dtypes.
- All 153 selected GEMM owners/2448 calls and 13 observed contracts pass together; other sixteen non-GEMM adapters remained native in this run. This is not all model compute/ABI support.
- 153 full weight byte scans at four phase boundaries; each call checks owner/storage/shape/stride/dtype/pointers/version metadata. NOT per-call full W byte preservation. A transient W change undone before a boundary is not ruled out by boundary hashes alone.
- All 2448 observed call weight versions were available, but inference version absence remains an explicitly handled general limitation.
- All shadow outputs are raw/strict/finite exact; replacement X/output hashes are linked to same-run native shadow. Public logprobs cover 2560 observed values, not all full-vocabulary values.
- Default BF16 reduced precision and split-K flags True/True remained unchanged. Prior public logger observed D99/math0/compute68; this combined run does not log/reassert hidden algorithm internals or precision across arbitrary cuBLAS versions.
- Correctness-only instrumentation. No timing improvement, kernel share, 100% eager GPU event coverage, graph/optimized-runtime equivalence, or universal numerical equality claim.
- Recovery facts are from the independently saved recovery at its timestamp, not a fresh GPU query by this audit.

## Evidence

- `allGemmIntegration01/RESULT.json`: `e45f5f61b355a1e0d4c4d8b5192f314ca55e3ad0cf96b5be5e2e084dcf4293aa`
- `allGemmIntegration01-launch.json`: `62ab5051e65bf4c251c5ec366561f3f0c743ac0b1757314b8d7fb9f8259611b9`
- `allGemmIntegration01-recovery.json`: `db949768b7b0e3fb53f4572ef351ffb58102b5ce91734d0d1f6ec38df218c5ac`
- `PINS.json`: `c0436b63c358063d486e9e48718b57ba0e8a374e9c975d340f90882bc40bdc8a`
- `CACHE-PRESEED-01.json`: `242fc6338f4b3d263de5d30e5620bcb390bbfd57d1cd013f91c47155af91a7a8`

Prior sample results, failures, ba360 stronger per-call W proof, and all frozen runtime sources are unchanged. This report and FINAL-COVERAGE.json are append-only follow-up artifacts.
