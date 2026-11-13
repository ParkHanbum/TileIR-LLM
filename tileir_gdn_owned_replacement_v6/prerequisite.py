"""Read-only prior actual shadow proof; no cache inspection."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent.parent/'tileir_gdn_owned_qualification_v5'
PINS={'gdnOwnedShadow01/RESULT.json':'881474d2bdb61a4779f195e54143753b27a5b5359657226d575436c9f687401d','gdnOwnedShadow01-launch.json':'57d7688def8a681ea11cb1e4b45b2952085f637ecb97ff0de26389ebfd2a0ca3','gdnOwnedShadow01-recovery.json':'bbf14a637d1f4e4e6b91b9dc02010e4ea49fff6809c15344ddc1eac0f1193084'}
def verify_shadow():
    docs=[]
    for name,h in PINS.items():
        data=(R/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==h,'prior shadow pin'
        docs.append(json.loads(data))
    r,l,z=docs
    assert r['complete'] and l['returncode']==z['actual_returncode']==0
    assert z['child_absent'] and not z['remaining_process_group'] and z['lock_available']
    assert l['result_sha256']==z['result_sha256']==PINS['gdnOwnedShadow01/RESULT.json']
    assert z['launch_sha256']==PINS['gdnOwnedShadow01-launch.json']
    a=r['adapter_install'][0]['owned_shadow']['synthetic'];assert a['passed'] and len(a['cases'])==6
    raw=r['lean_request']['selected_dispatch']['owned_shadow']['raw']
    assert raw['native_returns']==raw['candidate_returns']==raw['passed']==1152 and not raw['failed']
    return dict(passed=True,prior_actual_shadow=True,synthetic_cases=6,model_raw=1152,evidence=PINS)
