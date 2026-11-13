"""Bind actual full component proofs; new combined shadow remains mandatory."""
from pathlib import Path
import json,hashlib,importlib
from public_vllm_composed_integration_v5.gates import public_logprobs,compare_public

def qualification(root):
 p=Path(root)/'FULL-PROOFS.json';assert p.is_file(),'Missing full proofs'
 proofs=json.loads(p.read_text());assert [x['module'] for x in proofs]==['public_vllm_composed_integration_v8','tp1_embedding_runtime_integration_v1','attention_kv_runtime_integration_v2'];assert len(proofs)==3,'Three actual full proofs required'
 for item in proofs:
  for key in ('result','recovery','launch'):
   assert hashlib.sha256(Path(item[key]).read_bytes()).hexdigest()==item[key+'_sha256']
  r=json.loads(Path(item['result']).read_text());rec=json.loads(Path(item['recovery']).read_text());launch=json.loads(Path(item['launch']).read_text())
  assert importlib.import_module(item['module']+'.result_gate').audit(r)['passed']
  assert r['public_logprob_gate']['exact_numeric'] and r['public_logprob_gate']['max_abs']==0 and r['public_logprob_gate']['values']==2560
  assert launch['returncode']==0 and rec['child_pid_absent'] and rec['lock_available']
  for arm in ('shadow_request','replacement_request'):
   assert r[arm]['allow_bf16_reduced_precision_reduction'] is True and r[arm]['allow_bf16_reduced_precision_reduction_split_k'] is True
 return dict(proofs=proofs,scope='Separate actual full proofs; new combined2808 shadow still required')
