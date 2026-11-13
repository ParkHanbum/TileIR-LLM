"""Exact completed prerequisites only; new composition still needs execution."""
import importlib,json,hashlib
from pathlib import Path
def verify():
 refs=json.loads(Path(__file__).with_name('ACTUAL-PREREQUISITES.json').read_text())
 assert set(refs)=={'whole_v4','post16','sampling144','temperature16','integer72','whole_v5','joint_pair_copy'}
 for label,row in refs.items():
  data={}
  for k,v in row['evidence'].items():
   b=Path(v['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==v['sha256'];data[k]=json.loads(b)
  d,l,z=data['result'],data['launch'],data['recovery']
  assert importlib.import_module(row['audit_module']).audit(d,Path(row['root']))['passed']
  assert l['returncode']==z['actual_returncode']==0 and l['result_gate']['passed']
  assert z['child_absent']and not z['remaining_process_group']and z['lock_available']
  assert l['result_sha256']==z['result_sha256']==row['evidence']['result']['sha256']
 return dict(actual_prior_domains=list(refs),new_whole_endpoint_pending=True)
