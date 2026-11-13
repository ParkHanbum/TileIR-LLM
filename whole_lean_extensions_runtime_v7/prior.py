"""Historical raw/model proof gates; original v3 outer1 remains outer1."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent.parent

def read(path,sha):
 b=Path(path).read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
def recovered(r,h):
 assert r['actual_returncode']==0 and r['child_absent'] and not r['remaining_process_group'] and r['lock_available'] and r['result_sha256']==h

def verify():
 R=P/'whole_lean_extensions_runtime_v3'
 receipt=read(R/'independent_direct_verify_01/RECEIPT.json','7c5a07d67234e481ea7dab13c5a595503b5863d0d5450b906f97ab9907dfbda0')
 assert receipt['actual_model_child_exit']==0 and receipt['original_model_outer_exit']==1 and receipt['explicit_root_passed']is True and receipt['default_root_passed']is False
 assert len(receipt['negative_controls'])==4 and all(v is False for v in receipt['negative_controls'].values())
 artifacts={}
 for path,h in receipt['artifact_sha256'].items():
  b=Path(path).read_bytes();assert hashlib.sha256(b).hexdigest()==h
  if path.endswith('.json'):artifacts[Path(path).name]=json.loads(b)
 result=artifacts['RESULT.json'];launch=artifacts['wholeLeanExtensions03-launch.json'];recovery=artifacts['wholeLeanExtensions03-recovery.json']
 assert launch['returncode']==0 and launch['result_gate']['passed']is False
 recovered(recovery,'880b03f5d8254a9d85dced1504c89707ce2a8bd2d64c0dfbc0680885b6fe9679')
 from whole_lean_extensions_runtime_v3.gates import audit
 assert audit(result,R)['passed']
 from lean_finite_504_v1.proof import verify as finite
 f=finite()
 refs=json.loads((P/'lean_eq408_v1/PREREQUISITE.json').read_text())['evidence']
 q={k:read(v['path'],v['sha256'])for k,v in refs.items()}
 from eq408_runtime_v1.gates import audit as eq_audit
 assert eq_audit(q['result'],P/'eq408_runtime_v1')['passed']
 assert q['launch']['returncode']==0 and q['launch']['result_sha256']==refs['result']['sha256']
 recovered(q['recovery'],refs['result']['sha256'])
 return dict(prior_whole_explicit_root_audit=True,prior_child_exit=0,prior_outer_exit=1,prior_original_launcher_gate=False,finite504=f,eq408=True,new_combined_endpoint_pending=True)
