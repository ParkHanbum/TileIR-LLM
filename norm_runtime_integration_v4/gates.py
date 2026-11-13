import json,hashlib
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 receipt=Path(root)/'QUALIFICATION-RECEIPT.json'
 assert receipt.exists(),'Pending saved two-case actual PASS and independent recovery binding'
 evidence=json.loads(receipt.read_text())
 for p,h in evidence['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 r=json.loads(Path(evidence['result']).read_text());l=json.loads(Path(evidence['launch']).read_text());c=json.loads(Path(evidence['recovery']).read_text())
 assert r['complete'] and r['gate']['raw_pass'] and l['returncode']==0 and c['child_pid_absent'] and c['lock_available']
 return evidence
