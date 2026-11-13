"""Prior combined actual proof, verified read-only; no runtime payload oracle."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent/'residual_four_gdn_owned_runtime_v1'
PINS={'residualFourGDNOwned01/RESULT.json':'4cf0891a97db0fd32e69a0822eca09f1ddc4470aaf4db9156af6d5505dcc507f','residualFourGDNOwned01-launch.json':'1b51b39ff56f148a03a21c65c0cb61aeb71ba16cbe1bd5c00dff26556fb08abb','residualFourGDNOwned01-recovery.json':'9d15a4e9be1935f70b085777e752ce9beca1e3701dc70c42df5c5efe786d2431'}
def verify_shadow():
    docs=[]
    for name,h in PINS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h;docs.append(json.loads(raw))
    r,l,z=docs
    assert r['complete'] and l['returncode']==z['actual_returncode']==0 and l['qualification_complete']
    assert z['child_absent'] and not z['remaining_process_group'] and z['lock_available']
    assert z['result_sha256']==l['result_sha256']==PINS['residualFourGDNOwned01/RESULT.json']
    assert z['launch_sha256']==PINS['residualFourGDNOwned01-launch.json']
    f=r['four_replacement'];assert f['complete'] and f['restored'] and len(f['rows'])==64 and all(x['passed'] and x['candidate_calls']==1 and x['native_calls']==0 for x in f['rows'])
    g=r['lean_request']['selected_dispatch']['owned_replacement'];v=g['raw']
    assert v['passed']==v['candidate_returns']==v['cpu_oracle_passes']==1152 and v['native_target_calls']==0
    assert g['original_bindings_restored'] and g['core_bindings_unchanged']
    return dict(passed=True,prior_combined_actual=True,evidence=PINS,raw_repeated_in_lean=False)
