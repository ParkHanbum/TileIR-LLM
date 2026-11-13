import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from packed_k128_qualification_v1.run import result_gate

def qualification(root):
 e=json.loads((Path(root)/'QUALIFICATION-RECEIPT.json').read_text());docs={}
 for k,v in e.items():
  p=Path(v['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'];docs[k]=json.loads(p.read_text())
 assert result_gate(docs['result'])['raw_pass'] and docs['launch']['returncode']==docs['recovery']['actual_returncode']==0
 assert docs['recovery']['child_absent']and docs['recovery']['lock_available']
 return dict(scope='K128 two applicable actual samples qualified; live360 calls remain unqualified',result_sha256=e['result']['sha256'])
