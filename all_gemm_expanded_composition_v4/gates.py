"""Nine separate actual-exit0 proofs including parent, not the interrupted old19 run."""
from pathlib import Path
import hashlib,json
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from all_gemm_sixteen_sampler_composition_v2.result_gate import audit as base_audit
from sampler_four_runtime_integration_v1.result_gate import audit as sampler_audit
from gdn_prefill_core_runtime_integration_v3.result_gate import audit as gdn_audit
from attention_scoped_runtime_v5.result_gate import audit as attention_audit
from runner_padding_runtime_integration_v1.result_gate import audit as padding_audit
from runner_hidden_gather_runtime_integration_v2.result_gate import audit as gather_audit
from sampler_fp32_copy_runtime_integration_v1.result_gate import audit as cast_audit
from mamba_accepted_runtime_integration_v2.result_gate import audit as accepted_audit
AUDITS=dict(base18=base_audit,sampler_four=sampler_audit,gdn_prefill_core=gdn_audit,attention_core=attention_audit,runner_padding=padding_audit,hidden_gather=gather_audit,sampler_fp32=cast_audit,accepted_scatter=accepted_audit)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def qualification(root):
 from .parent_contract import verify_qualified_parent
 parent_proof=verify_qualified_parent(root)
 e=json.loads((Path(root)/'QUALIFICATION-RECEIPT.json').read_text());assert set(e)==set(AUDITS)
 detail={}
 for key,gate in AUDITS.items():
  row=e[key];loaded={}
  for kind in ('result','launch','recovery'):
   ref=row[kind];assert sha(ref['path'])==ref['sha256'];loaded[kind]=json.loads(Path(ref['path']).read_text())
  r,l,z=(loaded[k] for k in ('result','launch','recovery'))
  assert gate(r)['passed'],key
  assert l['returncode']==z['actual_returncode']==0 and l.get('child_absent') and z['child_absent']
  assert z.get('lock_available',z.get('lock_reacquired')) and not z.get('remaining_process_group')
  assert l['result_sha256']==z['result_sha256']==row['result']['sha256']
  assert not l.get('timeout') and not l.get('error')
  if 'launch_sha256' in z:assert z['launch_sha256']==row['launch']['sha256']
  if 'launcher_sha256' in z:assert z['launcher_sha256']==row['launch']['sha256']
  if 'final_seal' in row:assert sha(row['final_seal']['path'])==row['final_seal']['sha256']
  detail[key]=dict(passed=True,actual_exit=0,evidence=row)
 return dict(scope='18-controller actual proof plus eight independent standalone actual-exit0 proofs including parent; not combined25 proof; old19 exit-unobserved excluded',evidence=detail,parent_actual=parent_proof)
