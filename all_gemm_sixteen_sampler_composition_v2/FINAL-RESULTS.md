# Combined all-GEMM +16 adapters +3-way sampler: PASS

One combined native-shadow request and one conditional replacement request passed: **5,680 selected logical operations / 5,648 helper-GEMM boundaries each**, across18 controllers. All selected native/fallback paths are absent in replacement; selected fallback0 does not mean every model operation is replaced. All13 GEMM contracts/153 owners/2,448 calls pass together with the sixteen adapters and Top5/ranks/ordered selected logprobs. Frozen512 token IDs and2,560 public logprob values match exactly, defaultTrue/True preserved.

Child595599 and launcher actually exited0. Saved independent recovery confirms child/process-group absent, lock available, and existing GUI identities retained. This CPU audit independently rehashed1,912 pins and1,033 original cache files unchanged, verified receipt destination/hash linkage, all18 reverse closes and failure-free gates. Original proofs and numerical sources are unchanged.

| Selected component | Helper/GEMM boundaries | Logical operations |
|---|---:|---:|
| norm | 1040 | 1040 |
| attention_gate | 128 | 128 |
| block_gather | 16 | 16 |
| slot_mapping | 16 | 16 |
| pos_seq | 16 | 16 |
| gated_norm | 384 | 384 |
| sample_counts | 16 | 16 |
| qk_rope | 128 | 128 |
| silu_mul | 512 | 512 |
| indexed_prefill_conv | 24 | 24 |
| indexed_decode_conv | 360 | 360 |
| post_conv | 24 | 24 |
| embedding | 16 | 16 |
| attention_kv | 128 | 128 |
| greedy_sample | 16 | 16 |
| gdn_decode_core | 360 | 360 |
| all_gemm | 2448 | 2448 |
| sampler_three | 16 | 48 |

## Limits

- B32/P53/G16 eager BF16 TP1,32 identical prompts, one combined shadow/replacement execution. Not general diverse-input support.
- 5680 is selected logical operations;5648 is enclosing helper/GEMM boundaries. Neither is a GPU launch count or all-model denominator.
- Selected vendor fallback is0; attention core/GDN prefill/native cat and uncensused runner/device operations remain. Not100% model compute support.
- All-GEMM full output raw proof covers2590441472 BF16 elements in shadow and hashes them against replacement. Other components retain their existing alias/state/selected-row/compact-state proof scopes; no universal full-state invariance claim.
- GEMM153 W tensors are full-byte checked at4 phase boundaries and metadata/version checked each call; not per-call full W byte preservation.
- Public2560 logprob values are exact. Sampler helper selected scores/top5/ranks gates pass; this does not mean full-vocabulary output equality for every non-GEMM operation.
- Shadow495.543s and replacement49.849s contain heavy correctness validation. Their ratio is not a speedup. No production-like latency/throughput/cuTile share has been measured by this run.
- Peak reserved15378415616B and allocated14605186560B are PyTorch accounting, not proof of arbitrary future fit or total device VRAM usage. No runtime/KV settings were lowered.
- Recovery is the separately saved post-run observation, not a fresh GPU query in this CPU audit.

## Evidence SHA256

- `allGemmSixteenSampler02/RESULT.json`: `c45d43017d5596ef12aa2e58ca1d8c29374bbfc650ca0380aaac55bd74051636`
- `allGemmSixteenSampler02-launch.json`: `ceae0db8c271730edaff4681cf29400d9eed57a573de3dcf0768b64ce9749b52`
- `allGemmSixteenSampler02-recovery.json`: `3f14da626a6b40b023b9a712941047b25240f1445a71cc7d58f66a42574c0ef2`
- `PINS.json`: `0c25c102c536f8ecef70f9e97d4242c6e33b27d271d79789ca5dc08bb1f619c0`
- `CACHE-PRESEED-01.json`: `4315d08022f558b2497a5706e34fe7440dcb72cbba3f32cd6212e1ebdbf34aba`
