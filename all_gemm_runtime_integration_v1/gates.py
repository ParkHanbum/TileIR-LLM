from pathlib import Path
import hashlib,json
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from packed_k128_runtime_integration_v1.result_gate import audit as ba_audit
from unpacked_gemm_qualification_v1.contract import result_gate as five_gate

def qualification(root):
 e=json.loads((Path(root)/'QUALIFICATION-RECEIPT.json').read_text());d={}
 for k,v in e.items():
  p=Path(v['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'];d[k]=json.loads(p.read_text())
 assert ba_audit(d['ba_live'])['passed']and five_gate(d['five_sample'])['raw_pass']
 for name in ('ba','five'):
  assert d[name+'_launch']['returncode']==d[name+'_recovery']['actual_returncode']==0 and d[name+'_recovery']['child_absent']and d[name+'_recovery']['lock_available']
 return dict(scope='ba360 full selected proof plus five isolated contracts; all2448 combined proof remains pending',evidence=e)
