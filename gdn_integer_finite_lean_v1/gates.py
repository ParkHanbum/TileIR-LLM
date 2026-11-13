import hashlib,json
from pathlib import Path
from gdn_integer_finite_runtime_qualification_v3.gates import audit

def qualification():
 refs=json.loads(Path(__file__).with_name('PREREQUISITE.json').read_text());e={}
 for k,v in refs.items():
  p=Path(v['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'];e[k]=json.loads(p.read_text())
 assert audit(e['result'])['passed']
 l,z=e['launch'],e['recovery'];assert l['returncode']==z['actual_returncode']==0 and l['child_absent'] and z['child_absent'] and not l['remaining_process_group'] and not z['remaining_process_group'] and l['lock_reacquired'] and z['lock_available']
 assert l['result_sha256']==z['result_sha256']==refs['result']['sha256']
 return {'actual_integer72_raw_proof':True,'lean_endpoint_pending':True}
