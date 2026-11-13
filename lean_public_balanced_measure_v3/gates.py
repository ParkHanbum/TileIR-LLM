import hashlib,json,math
from pathlib import Path
from lean_public_runtime_qualification_v2.gates import sha,exact_tokens,exact_public,dispatch,frozen_reference
from lean_public_composition_v2.adapter import ORDER
from .design import orders,profile_order,validate_schedule
from .profile_contract import inspect_trace,scope_names
P=Path(__file__).resolve().parent.parent
Q=P/'lean_public_runtime_qualification_v2'
def qualification(root):
 from lean_public_runtime_qualification_v2.gates import audit as correctness_audit
 refs=json.loads((Path(root)/'QUALIFICATION-RECEIPT.json').read_text())
 assert refs.get('complete') is True,'Lean25 actual correctness pending; no GPU/admission/cache allowed'
 assert refs['qualification_package']==Q.name
 data={}
 for key in ('result','launch','recovery','source_pins'):
  ref=refs[key];path=Path(ref['path']);assert path.is_absolute() and sha(path)==ref['sha256'],key
  if key=='result':assert path.parent.parent==Q
  else:assert path.parent==Q
  data[key]=json.loads(path.read_text())
 assert correctness_audit(data['result'],Q)['passed'],'Lean25 actual output/dispatch/restore gate'
 result_sha=refs['result']['sha256'];l,z=data['launch'],data['recovery']
 assert l['returncode']==z['actual_returncode']==0 and l['finished'] and not l.get('timeout') and not l.get('error')
 assert l['result_sha256']==z['result_sha256']==result_sha and z['launch_sha256']==refs['launch']['sha256']
 assert l['child_absent'] and z['child_absent'] and not l.get('remaining_process_group') and not z.get('remaining_process_group')
 assert l.get('lock_available',l.get('lock_reacquired')) and z.get('lock_available',z.get('lock_reacquired'))
 assert z['source_pin_file_sha256']==refs['source_pins']['sha256']
 for path,h in data['source_pins'].items():assert sha(path)==h,('Changed lean25 source',path)
 assert z['source_pins_verified']==len(data['source_pins'])
 assert z['baseline_cache_files_verified']==1033
 return dict(evidence=refs,scope='Completed25-controller lean endpoint proof; full25 per-component raw/contract proof separate; old19 exit-unobserved excluded')
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
