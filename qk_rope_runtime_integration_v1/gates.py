import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 q=P/'qk_rope_qualification_v1'; r=q/'qkRopeQual01/RESULT.json'; c=q/'qkRopeQual01-recovery.json'; l=q/'qkRopeQual01-launch.json'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='014d6a40481592f716608cbe607763b7f87906d765e680f9815bd83ed2c556a7'
 assert hashlib.sha256(c.read_bytes()).hexdigest()=='57ecd0e7bd0e8857f7c873b73f5e91bca82537ec9ea586ed3d274441ae3b2ea5'
 assert json.loads(r.read_text())['gate']['raw_pass'] and json.loads(l.read_text())['returncode']==0
 rec=json.loads(c.read_text());assert rec['child_pid_absent'] and rec['lock_available']
 return dict(scope='Two layer3 actual cases; all128 shadow calls required before replacement',result_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
