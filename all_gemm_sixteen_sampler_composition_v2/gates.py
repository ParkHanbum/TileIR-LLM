from pathlib import Path
import hashlib,json
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from all_gemm_runtime_integration_v1.result_gate import audit as gemm_audit
from public_vllm_composed_integration_v11.result_gate import audit as sixteen_audit
from sampler_three_ordered_runtime_integration_v2.result_gate import audit as top5_audit

def qualification(root):
 e=json.loads((Path(root)/'QUALIFICATION-RECEIPT.json').read_text());d={}
 for k,v in e.items():
  p=Path(v['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'];d[k]=json.loads(p.read_text())
 for name,gate in (('gemm',gemm_audit),('sixteen',sixteen_audit),('sampler',top5_audit)):
  assert gate(d[name+'_result'])['passed']
  l,z=d[name+'_launch'],d[name+'_recovery'];assert l['returncode']==z['actual_returncode']==0 and z['child_absent']and z['lock_available']
  assert l['result_sha256']==z['result_sha256']==e[name+'_result']['sha256']
 return dict(scope='Three separate proofs; combined composition pending. Prior sampler v2 fresh-cache proof linked; this composition still requires its own destination-bound receipt.',evidence=e)
