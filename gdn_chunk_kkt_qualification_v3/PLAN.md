# KKT K8 candidate replay

Candidate tileir_gdn_chunk_kkt_v3 changes only MMA granularity: ascending K8 over16 blocks; beta multiplication, RZ conversion, exp factor and masks unchanged. It restricts provenance to the previous actual native BK64/warps8/stages3. Prior diagnostics established same beta*k and exp factor but not the exact remaining cause. K8 is a new hypothesis, not a claimed fix.

Reuse the exact saved native_reference.pt from v1's successful native2 calls and identical saved core inputs/native cumsum, as in v2 qualifier. No fresh native call or autotune and no claim of same-run native comparison. Candidate twice; stop first raw/strict/finite/repeat/storage/alias/+0mask failure. All existing gates and tolerances unchanged. Parent-only GPU after independent review, exclusive lock/fresh admission/285s child/300s outer/recovery. No performance, full24/full-core/model or adoption claim.
