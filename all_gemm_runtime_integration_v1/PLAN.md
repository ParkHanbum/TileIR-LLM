# All13 GEMM contracts: combined native-shadow plan

CPU plan only. No new model, GPU query, compile or cache operation. The new run leaves all16 non-GEMM adapters disabled/native.

178 modules own apply, but25 are not GEMM:24 convolution modules own3D weights consumed directly by public GDN convolution code, and embed_tokens uses embedding/F.embedding. Actual153 GEMM owners are88 packed and65 unpacked, each16 calls =2,448. This exactly reconciles prior native packed census1,408 + unpacked census1,040 and official cuBLAS D logs2,448/13 shape-ld contracts. LM head is M32 in both phases; phase must follow per-owner call index.

## Combined proof

1. Match all178 live owners to the pinned153-selected/25-excluded inventory, including names, shape/stride, source and unique method ownership. Install process-local public apply wrappers. Add fail-closed guards for unexpected apply on25 excluded owners, while keeping their actual convolution/embedding APIs original. Restore all changed bindings.
2. First request returns each original native output object and compares same-input TileIR for all2,448 calls. K128 only for ba decode360; unchanged existing TileIRBLAS for2,088 others. Raw/strict/finite/layout/alias/input-preservation per call. First discrepancy retained, replacement forbidden.
3. Only complete153-owner/13-contract coverage permits second request:2,448 selected TileIR applies, native/fallback0. Require corresponding X/W/output hashes to match verified native shadow; preserve packing/layout. Other operations stay original.
4. Fixed512 IDs and public2,560 logprobs exact, original T/T read-only, same53x32/G16 eager async workload, source/cache preservation, binding restoration, actual exit and independent recovery. No performance claim.

## Verification cost

To retain full-byte checks, hash full weight data in16MiB CPU chunks before/after calls, without persistent weight dumps. The exact conservative byte-scan bound is in PLAN-INVENTORY.json and may dominate runtime. Input/output comparisons should also use bounded chunks. Version/pointer/stride fingerprints supplement byte hashes, not silently replace them. Save only first mismatch within a fixed cap. Parent should review a total time bound before GPU execution; runtime is not measured yet. Preparation can continue without GPU.

Old7 packed sample limitations are resolved only by successful new default-mode comparisons, not by relabeling old evidence. All13 GEMM support is not all GPU-op support or performance parity.
