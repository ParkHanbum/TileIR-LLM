# Native five-contract capture: verified completion

Native-only capture passed, not a TileIR qualification. Original v1 failure is preserved.

- 65 owners × 16 calls = 1,040 exact-once native calls; 8 representative banks, 118,816,768 logical bytes.
- Four checkpoint named tensors matched live weight bytes and final hashes. No duplicated weight file.
- Both requests match 512 fixed token IDs; 2,560 public logprob values exactly agree. Original True/True flags unchanged.
- Child/launcher exit 0 and independent recovery confirm child absence/common lock availability. 59 source pins verified unchanged. No performance claim.

| bank | logical M,N,K | API lda,ldb,ldc |
|---|---|---|
| gdn_out_prefill | 1696,2560,4096 | 4096,4096,2560 |
| down_prefill | 1696,2560,9216 | 9216,9216,2560 |
| attn_out_prefill | 1696,2560,4096 | 4096,4096,2560 |
| head_prefill | 32,248320,2560 | 2560,2560,248320 |
| gdn_out_decode | 32,2560,4096 | 4096,4096,2560 |
| down_decode | 32,2560,9216 | 9216,9216,2560 |
| attn_out_decode | 32,2560,4096 | 4096,4096,2560 |
| head_decode | 32,248320,2560 | 2560,2560,248320 |

All eight: public cublasGemmEx, column-major T/N, BF16 A/B/C, compute32F=68, requested algorithm99, math0. Exact dimensions/leading dimensions/A-B-C pointers identify one native apply per log. Alpha1/beta0 derive from no-bias Linear source; logger gives pointers only. Pointer mode and internal selected algorithm/partial precision are unobserved.

Coverage remains 7 sampled exact packed contracts, 1 sampled packed mismatch, 5 candidate-unverified contracts now backed by actual native captures. A new qualification is needed; no older TileGym proof fills this gap.

## Evidence
- result: `/home/n/work/TileIR/public-tileir-blas-20261111/unpacked_gemm_native_capture_v2/fiveGemmCapture02/RESULT.json` SHA256 `33b58427247f7c5e2446371c3ae1d5bf7e23453333feb2c0220547992f371fff`
- launch: `/home/n/work/TileIR/public-tileir-blas-20261111/unpacked_gemm_native_capture_v2/fiveGemmCapture02-launch.json` SHA256 `e0e3e5c773967c30f3d982f365989ed18355daea3f56c825b01d289f311c7a36`
- recovery: `/home/n/work/TileIR/public-tileir-blas-20261111/unpacked_gemm_native_capture_v2/fiveGemmCapture02-recovery.json` SHA256 `7e74b54e8aedb769ad2f47d53e040751bb18330c049a5f64b0f71648083c55b4`
- manifest: `/home/n/work/TileIR/public-tileir-blas-20261111/unpacked_gemm_native_capture_v2/fiveGemmCapture02/bank/MANIFEST.json` SHA256 `8cea9b1d84ce64b87a752a53a649066c2492260bc68bce01c26051de7b8a7346`
