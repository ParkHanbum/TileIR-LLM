import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 p=P/'slot_mapping_qualification_v1';r=p/'slotMappingQual01/RESULT.json';c=p/'slotMappingQual01-recovery.json';l=p/'slotMappingQual01-launch.json'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='3c6f95ac47af3c2a739ac71a534923a5965d653b7e9ec07d6b7cc583e96ff203'
 assert hashlib.sha256(c.read_bytes()).hexdigest()=='a902cf0059d43f6e5a6b77f3bb33c2c8fd72750b620cf0bfb9e1ac9c3e8c832c'
 assert json.loads(r.read_text())['gate']['raw_pass'] and json.loads(l.read_text())['returncode']==0
 rec=json.loads(c.read_text());assert rec['child_pid_absent'] and rec['lock_available']
 return dict(scope='Two frozen actual cases full-capacity slots qualified; all16 native shadow calls still required',result_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
