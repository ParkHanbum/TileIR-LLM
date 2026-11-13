"""Verified final-v7 endpoint prerequisite; never performs admission or GPU work."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
V5=R.parent/'whole_lean_extensions_runtime_v7'
PINS_SHA='86f60d89027d4649cee1e9d7d11eb77f10e3eaea4556ac463f89407db05c9789'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_documents(d,l,z,expected_result,expected_launch):
    assert l['returncode']==0 and z['actual_returncode']==0
    assert l['result_gate']['passed'] is True and l['qualification_complete'] is True
    assert l['child_absent'] is True and z['child_absent'] is True
    assert not l['remaining_process_group'] and not z['remaining_process_group']
    assert l['lock_reacquired'] is True and z['lock_available'] is True
    assert 'after' in l
    assert not any(l.get(k) for k in ('error','timeout','recovery_error'))
    assert z['source_cache_unchanged'] is True and z['pins_checked']==6549
    assert l['result_sha256']==z['result_sha256']==expected_result
    assert z['launch_sha256']==expected_launch
    assert d['complete'] is True and d['candidate_checks_passed'] is True

def require_ready():
    receipt=json.loads((R/'ACTUAL-ENDPOINT-PREREQUISITE.json').read_text())
    assert receipt['status']=='actual_endpoint_verified'
    assert receipt['root']==str(V5) and receipt['source_pins_sha256']==PINS_SHA
    assert sha(V5/'PINS.json')==PINS_SHA
    docs={}
    for label in ('result','launch','recovery'):
        ref=receipt['evidence'][label];assert sha(ref['path'])==ref['sha256']
        docs[label]=json.loads(Path(ref['path']).read_text())
    verify_documents(docs['result'],docs['launch'],docs['recovery'],receipt['evidence']['result']['sha256'],receipt['evidence']['launch']['sha256'])
    # Public CPU audit uses the original actual root explicitly, never profile root.
    from whole_lean_extensions_runtime_v7.gates import audit
    assert audit(docs['result'],V5)['passed'] is True
    return dict(passed=True,actual_endpoint=True,worker_exit0=True,outer_gate_passed=True,recovered=True,evidence=receipt['evidence'],profile_gpu_executed=False)
