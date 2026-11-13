import json
from pathlib import Path
from runner_scalar32_qualification_v3.gates import sha,frozen_reference,exact_tokens,exact_public
from runner_scalar32_qualification_v3.gates import audit as raw_audit
def qualification(root):
 root=Path(root)
 for p,h in json.loads((root/'PINS.json').read_text()).items():assert sha(p)==h,p
 q=json.loads((root/'PREREQUISITE.json').read_text());assert q['complete'] is True
 data={}
 for k,v in q['evidence'].items():
  assert sha(v['path'])==v['sha256'];data[k]=json.loads(Path(v['path']).read_text())
 assert raw_audit(data['result'])['passed']
 l,z=data['launch'],data['recovery']
 assert l['returncode']==z['actual_returncode']==0 and l['child_absent'] and z['child_absent'] and not l['remaining_process_group'] and not z['remaining_process_group']
 assert l['lock_reacquired'] and z['lock_available'] and l['result_sha256']==z['result_sha256']==q['evidence']['result']['sha256']
 return dict(raw_scalar_actual_prerequisite=True,lean_endpoint_pending=True)
def audit(r,root=None):
 try:
  root=Path(root or Path(__file__).parent)
  assert r['complete'] is True and r['candidate_checks_passed'] is True and not r.get('error') and not r.get('cleanup_errors')
  ft,fp,options=frozen_reference();assert r['engine_options']==options
  for mode in ('native','lean'):
   x=r[mode+'_request'];exact_tokens(x['output_token_ids'],ft);exact_public(x['public_logprobs'],fp)
   assert x['allow_bf16_reduced_precision_reduction'] is True and x['allow_bf16_reduced_precision_reduction_split_k'] is True
  assert r['lean_census']==dict(calls=32,attempts=32,indices=list(range(32)),failed=None,selected_native_calls=0,selected_fallback_calls=0,logical_operations=32)
  assert r['observer_restored'] and r['engine_shutdown'] and r['synchronized_after_restore'] and r['owner_binding_verified']
  assert r['precision_policy']['initial']==dict(bf16=True,split_k=True) and r['precision_policy']['restored'] and r['precision_policy']['restored_split_k']
  assert r['cache_preseed']['destination']==str(root/'cache_v1') and r['cache_preseed']['files_verified'] and r['cache_preseed']['separate_inodes_verified']
  return dict(passed=True,scope='lean scalar32 endpoint/census; raw proof separate prerequisite')
 except (AssertionError,KeyError,TypeError,ValueError) as e:return dict(passed=False,error=type(e).__name__)
