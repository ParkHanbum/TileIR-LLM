from pathlib import Path
from residual_four_gdn_owned_lean_v2.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch,audit as lean_audit

def qualification(root):
    # Only actual raw proofs are prerequisites; neither pending standalone lean is PASS.
    from residual_four_gdn_owned_lean_v2.prerequisite import verify_shadow
    from zero_kv_typed_lean_v1.gates import qualification as zero_raw
    return dict(combined_raw=verify_shadow(),zero_raw=zero_raw(Path(root).parent/'zero_kv_typed_lean_v1'),new_combined_lean_endpoint_pending=True)

def audit(r,root=None):
    a=lean_audit(r,root or Path(__file__).resolve().parent)
    if not a['passed']:return a
    try:
        d=r['zero_lean'];p=d['proof'];g=d['geometry']
        assert d['failed'] is None and not r.get('zero_cleanup_errors')
        assert d['setup_attempted']==d['setup_returned']==1
        assert d['native_reference_attempted']==d['native_reference_returned']==1
        assert d['fresh_registry_after_native']
        assert d['request_zero_attempted']==d['request_zero_returned']==0 and d['candidate_attempted']==d['candidate_returned']==1
        assert g['table_equal'] and g['readonly_tables_ids_exact']
        assert g['segments']==8 and g['allocations']==1 and g['unique_block_count']==g['block_count']==32 and g['selected_union_bytes']==553648128
        assert p['owner_payload_read_bytes']==p['owner_hashes_computed']==0 and p['target_oracle_executed'] is False and p['outside_oracle_executed'] is False
        assert p['zero_attempted']==p['zero_returned']==p['expected_source_launches']>0
        assert p['target_bytes']==553648128 and 2*p['allocation_bytes']-p['target_bytes']<=16*1024**3
        assert d['restored'] and d['released'] and r['engine_shutdown']
        return dict(passed=True,base=a,zero_request_candidates=1,zero_request_native=0,zero_target_bytes=553648128,cast16_integrated=False,performance=False)
    except (KeyError,TypeError,AssertionError) as e:return dict(passed=False,errors=[repr(e)])
