from pathlib import Path
from lean26_gdn_finite_runtime_v1.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch,audit as base_audit
from .prerequisite import verify_shadow

def qualification(root):return verify_shadow()

def audit(r,root=None):
    root=Path(root or Path(__file__).resolve().parent)
    a=base_audit(r,root)
    if not a['passed']:return a
    try:
        assert r['four_binding']==[dict(shared_latch=True,owned_copy_identity=True,per_helper_cap=16)]
        f=r['four_replacement'];assert f['complete'] and f['restored'] and not f['failed'] and f['cpu_oracle_calls']==0 and f['raw_rechecked'] is False
        assert f['calls']==dict(prepare_rope_positions=16,combine_sampled_and_draft_tokens=16,bias=16,post_update=16)
        assert all(x['passed'] and x['native_calls']==0 and x['candidate_calls']==1 and x['original_addresses_preserved'] and x['return_contract'] and x['preflight_passed'] for x in f['rows'])
        assert r['uva_constructor_restored'] and r['uva_release']['released'] and r['uva_release']['after_synchronization'] and not r.get('four_cleanup_errors')
        g=r['lean_request']['selected_dispatch']['owned_replacement'];v=g['operations']
        assert g['replacement_pass'] and g['original_bindings_restored'] and g['core_bindings_unchanged']
        assert v['calls']==v['candidate_returns']==v['metadata_passes']==1152 and v['native_target_calls']==0
        assert v['cpu_oracle_calls']==0 and v['raw_rechecked'] is False and v['returns_candidate'] and v['order_complete'] and not v['failed']
        return dict(passed=True,scope='New lean endpoints/operational guards/census only; prior raw proof separate',base_boundaries=5889,base_logical=6057,finite_guards=472,four=64,owned=1152,raw_rechecked=False)
    except (KeyError,AssertionError,TypeError)as e:return dict(passed=False,errors=[repr(e)])
