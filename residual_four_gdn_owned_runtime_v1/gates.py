import json
from pathlib import Path
from lean26_gdn_finite_runtime_v1.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch
from residual_four_lean26_runtime_v1.gates import audit as four_audit
from tileir_gdn_owned_replacement_v6.gates import audit as gdn_audit

def qualification(root):
    refs=json.loads((Path(root)/'PREREQUISITE.json').read_text())
    assert set(refs)=={'four_lean26','gdn_replacement'}
    for label,items in refs.items():
        data={}
        for kind,item in items.items():
            assert sha(item['path'])==item['sha256'];data[kind]=json.loads(Path(item['path']).read_text())
        r,l,c=data['result'],data['launch'],data['recovery']
        assert (four_audit(r,Path(items['result']['path']).parent.parent) if label=='four_lean26' else gdn_audit(r))['passed']
        assert l['returncode']==c['actual_returncode']==0 and l['qualification_complete']
        assert c['child_absent'] and not c['remaining_process_group'] and c['lock_available']
        assert l['result_sha256']==c['result_sha256']==items['result']['sha256'] and c['launch_sha256']==items['launch']['sha256']
    return dict(passed=True,evidence=refs,combined_actual_proof=False)

def audit(r,root=None):
    root=Path(root or Path(__file__).resolve().parent)
    a=four_audit(r,root)
    if not a['passed']:return a
    try:
        g=r['lean_request']['selected_dispatch']['owned_replacement']
        assert g['replacement_pass'] and g['original_bindings_restored'] and g['core_bindings_unchanged']
        v=g['raw'];assert v['calls']==v['candidate_returns']==v['cpu_oracle_passes']==v['passed']==1152
        assert v['native_returns']==v['native_target_calls']==0 and v['returns_candidate'] and not v['failed'] and v['order_complete']
        assert r['adapter_restore'][0]['owned_replacement']['closed']
        return dict(passed=True,base_boundaries=5889,base_logical_operations=6057,finite_guards=472,four_calls=64,gdn_owned_replacements=1152,scope='Separate actual censuses, not physical GPU coverage or speed')
    except (KeyError,AssertionError,TypeError) as e:return dict(passed=False,errors=[repr(e)])
