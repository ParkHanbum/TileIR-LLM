import hashlib,json,math
from pathlib import Path
from lean_public_runtime_qualification_v1.gates import sha,exact_tokens,exact_public,dispatch,frozen_reference
from lean_public_composition_v1.adapter import ORDER
from .design import orders,profile_order,validate_schedule
from .profile_contract import inspect_trace,scope_names
P=Path(__file__).resolve().parent.parent
Q=P/'lean_public_runtime_qualification_v1'
PROOF_SHA='5db76cc7712310d316942d5adcaa86f3fd4560ce464572a18318164801c6eebc'
def qualification(root):
 from lean_public_runtime_qualification_v1.gates import audit
 refs=json.loads((Path(root)/'QUALIFICATION-RECEIPT.json').read_text());d={}
 for k,v in refs.items():assert sha(v['path'])==v['sha256'];d[k]=json.loads(Path(v['path']).read_text())
 assert refs['result']['sha256']==PROOF_SHA and audit(d['result'],Q)['passed']
 l,z=d['launch'],d['recovery'];assert l['returncode']==z['actual_returncode']==0 and l['result_sha256']==z['result_sha256']==PROOF_SHA
 assert z['child_absent']and z['lock_available']and z['source_pins_unchanged']and z['original_cache_files_unchanged']and not z.get('remaining_process_group')
 return dict(evidence=refs,scope='Completed18-controller lean proof only; uncertain19-controller run not used')
def audit(r,root):
 try:
  assert r['complete']and not r.get('error')and not r.get('cleanup_errors')and r['candidate_checks_passed']and r['engine_shutdown']
  index=r['process_index'];assert index in(0,1)
  ft,fp,opts=frozen_reference();assert r['engine_options']==opts and r['baseline']=='current eager vLLM mode0/graphsNONE'
  c=r['cache_preseed'];assert c['destination']==str(Path(root)/'cache_v1')and c['files_verified']and c['separate_inodes_verified']
  p=r['precision_policy'];assert p['initial']==dict(bf16=True,split_k=True)and p['restored']and p['restored_split_k']
  validate_schedule(index,r['pairs']);assert[r['mode']for r in r['profile_requests']]==profile_order(index)
  assert[r['mode']for r in r['qualification_requests']]==['native','lean']
  requests=r['qualification_requests']+[x for pair in r['pairs']for x in pair['requests']]+r['profile_requests'];assert len(requests)==12
  for x in requests:
   exact_tokens(x['output_token_ids'],ft);exact_public(x['public_logprobs'],fp)
   assert x['allow_bf16_reduced_precision_reduction']is True and x['allow_bf16_reduced_precision_reduction_split_k']is True
   if x['mode']=='lean':
    dispatch(x['selected_dispatch']);assert x['adapter_install'][0]['install_order']==list(ORDER)
    z=x['adapter_restore'];assert len(z)==1 and z[0]['restored']and z[0]['close_order']==list(reversed(ORDER))and all(t['restored']for t in z[0]['components'].values())and x['adapter_release_absent']==[True]
   else:assert x['selected_dispatch']is None
  for pair in r['pairs']:
   for x in pair['requests']:assert x['measured']and math.isfinite(x['whole_ms'])and x['whole_ms']>0 and not x.get('profiled')
  for x in r['qualification_requests']+r['profile_requests']:assert not x['measured']and x['whole_ms']is None
  for x in r['profile_requests']:
   assert x['profiled']and x['trace_bytes']>0 and x['trace_bytes']<=128*1024*1024
   assert sha(x['trace_path'])==x['trace_sha256']
   assert x['profile_scope_names']==scope_names(x['mode'])
   partition=inspect_trace(x['trace_path'],x['mode']);assert partition==x['profile_partition']and partition['model_denominator_ready']
  assert r['final_adapter_absent']==[True]and r['profile_scope_restored']
  return dict(passed=True,process_index=index,pairs=4,wall_samples=8,profiles=2,scope='Warm-cache current-eager conditional comparison; public API guards remain; no100% claim')
 except(AssertionError,KeyError,TypeError,ValueError,IndexError)as e:return dict(passed=False,errors=[repr(e)])
