import json,hashlib
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 r=P/'attention_gate_runtime_qualification_v1/attentionGateQual01/RESULT.json';c=P/'attention_gate_runtime_qualification_v1/attentionGateQual01-recovery.json';l=P/'attention_gate_runtime_qualification_v1/attentionGateQual01-launch.json'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='27ecc207e8e54b6999ffc3d5ca3db6a0e868109ee838a863cb44d1cabcec9ec1'
 assert hashlib.sha256(c.read_bytes()).hexdigest()=='4ca21b3a2a361a67cfde9254d216cc9b67aee33b284b4e089d335b4b9211dbeb'
 data=json.loads(r.read_text());rec=json.loads(c.read_text());assert data['gate']['raw_pass'] and json.loads(l.read_text())['returncode']==0 and rec['child_pid_absent'] and rec['lock_available']
 return dict(scope='Representative layer3 two-case raw qualification only; all128 actual shadow calls required before replacement',result_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
