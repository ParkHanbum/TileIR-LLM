import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 p=P/'block_gather_qualification_v1';r=p/'blockGatherQual01/RESULT.json';c=p/'blockGatherQual01-recovery.json';l=p/'blockGatherQual01-launch.json'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='4191199caf6a3cf1923e512cde22538cb6b427d316c18a1376f75257dde8a646'
 assert hashlib.sha256(c.read_bytes()).hexdigest()=='3ed5cc3b529fdd70f2d70c150799aee346a795dc37abeb569be8a19f71d42765'
 assert json.loads(r.read_text())['gate']['raw_pass'] and json.loads(l.read_text())['returncode']==0
 rec=json.loads(c.read_text());assert rec['child_pid_absent'] and rec['lock_available']
 return dict(scope='Two frozen actual cases and mapped UVA qualified; all16 native shadow calls still required',result_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
