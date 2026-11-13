import json,hashlib
from pathlib import Path
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def qualification(root=None):
 r=Path(root or Path(__file__).parent)
 for p,h in json.loads((r/'PINS.json').read_text()).items():assert sha(p)==h
 q=json.loads((r/'PREREQUISITE.json').read_text());assert q['complete'] is True;e={}
 for k,v in q['evidence'].items():assert sha(v['path'])==v['sha256'];e[k]=json.loads(Path(v['path']).read_text())
 from runtime_metadata_five_v1.gates import audit
 assert audit(e['result'])['passed']
 assert e['launch']['returncode']==e['recovery']['actual_returncode']==0
 assert e['launch']['child_absent'] and e['recovery']['child_absent'] and not e['launch']['remaining_process_group'] and not e['recovery']['remaining_process_group']
 assert e['launch']['lock_reacquired'] and e['recovery']['lock_available']
 assert e['launch']['result_sha256']==e['recovery']['result_sha256']==q['evidence']['result']['sha256']
 return dict(qualified_raw5=True,actual_exit0=True,new_lean_endpoint_proven=False)
