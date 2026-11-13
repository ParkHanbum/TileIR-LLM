import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent.parent/'finite_504_runtime_v1'
FILES={'result':('finite504Model01/RESULT.json','b07c7546872b8449c0ed3ae9d812bd28bfb438eb968dc39d55f2dcafee0a206f'),'launch':('finite504Model01-launch.json','03965526d8fb1f09de6f7ba8d9a7dd3f8b9d21391316d2fa1aae79d52d12ad59'),'recovery':('finite504Model01-recovery.json','7affc98ced5d2cfe5489270e1fe7b7b498a37124c1cc32a4b9c26e1820183b3f')}
def verify():
 from finite_504_runtime_v1.gates import audit
 d={}
 for k,(name,h) in FILES.items():
  b=(R/name).read_bytes();assert hashlib.sha256(b).hexdigest()==h;d[k]=json.loads(b)
 assert audit(d['result'],R)['passed']
 assert d['launch']['returncode']==d['recovery']['actual_returncode']==0
 assert d['recovery']['child_absent'] and not d['recovery']['remaining_process_group'] and d['recovery']['lock_available']
 assert d['launch']['result_sha256']==d['recovery']['result_sha256']==FILES['result'][1]
 return dict(passed=True,result_sha256=FILES['result'][1])
