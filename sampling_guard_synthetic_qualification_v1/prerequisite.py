from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent.parent
LINKS={
'result':('i64_range_synthetic_qualification_v1/i64RangeSynthetic02/RESULT.json','b5b7e3d15dab85dcc8cd4caf428f36bad5ca351b7f6485ccc4ea6cf21577d59b'),
'launch':('i64_range_synthetic_qualification_v1/i64RangeSynthetic02-launch.json','2784fc98e55bf3f1036ea23bbcf2f4708d50153e422449d41f9b1ab2d6a20121'),
'recovery':('i64_range_synthetic_qualification_v1/i64RangeSynthetic02-recovery.json','65f2f7d38cf3fb97c7f69ec8b2a928ab36ac120fcdd7b727d01801b6728b36ab')}
def verify():
 docs={}
 for label,(rel,h)in LINKS.items():
  p=P/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;docs[label]=json.loads(p.read_text())
 from i64_range_synthetic_qualification_v1.run import result_gate
 assert result_gate(docs['result'])['raw_pass']
 l=docs['launch'];r=docs['recovery']
 assert l['returncode']==0 and l['result_gate']['raw_pass']and l['child_absent']and l['lock_reacquired']and not l.get('remaining_process_group')
 assert r['actual_returncode']==0 and r['child_absent']and r['lock_available']and not r.get('remaining_process_group')
 assert l['result_sha256']==r['result_sha256']==LINKS['result'][1]and r['launch_sha256']==LINKS['launch'][1]
 return dict(passed=True,evidence={k:dict(path=str(P/p),sha256=h)for k,(p,h)in LINKS.items()},scope='actual isolated32; not model48')
