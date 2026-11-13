from pathlib import Path
from lean26_gdn_finite_runtime_v1.gates import audit as base_audit

def audit(result):
    gate=base_audit(result,Path(__file__).resolve().parent);assert gate['passed'],gate
    proof=result['adapter_install'][0]['owned_replacement']['synthetic'];assert proof['passed'] and proof['prior_actual_shadow']
    replacement=result['lean_request']['selected_dispatch']['owned_replacement']
    assert replacement['replacement_pass'] and replacement['original_bindings_restored'] and replacement['core_bindings_unchanged']
    r=replacement['raw']
    assert r['calls']==r['candidate_returns']==r['passed']==r['cpu_oracle_passes']==1152
    assert r['native_returns']==r['native_target_calls']==0 and r['returns_candidate']
    assert not r['failed'] and r['order_complete']
    return dict(passed=True,base=gate,prior_shadow_required=True,replacement_operations=1152,native_targets_in_replacement_arm=0,cpu_byte_oracle=True)
