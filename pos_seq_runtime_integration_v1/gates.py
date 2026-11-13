import json,hashlib
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 r=P/'pos_seq_runtime_qualification_v1/posSeqQual01/RESULT.json';c=P/'pos_seq_runtime_qualification_v1/posSeqQual01-recovery.json';l=P/'pos_seq_runtime_qualification_v1/posSeqQual01-launch.json'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='758bc8c3b24bc96efcbaa65f882399253515f9e21386db91fad82efd0a4da635'
 assert hashlib.sha256(c.read_bytes()).hexdigest()=='e6aa13d78e81755ac211fbbfda9df38d1ae920101cd36ecc5ce62e17bb9be7f9'
 data=json.loads(r.read_text());rec=json.loads(c.read_text());assert data['gate']['raw_pass'] and json.loads(l.read_text())['returncode']==0 and rec['child_pid_absent'] and rec['lock_available']
 return dict(scope='Two actual full-storage cases only; full16shadow required before replacement',result_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
