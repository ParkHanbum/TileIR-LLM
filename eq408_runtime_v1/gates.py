import json
from pathlib import Path
from lean26_gdn_finite_runtime_v1.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch as base_dispatch,audit as base_audit
from lean26_gdn_finite_runtime_v1.adapter import ORDER
from eq408_runtime_v1.adapter import LIMITS

def eq_gate(c,mode):
 assert c['mode']==mode and c['calls']==408 and c['descale384']==384 and c['initial24']==24 and c['counts']==LIMITS
 assert c['native_predicates']==(408 if mode=='shadow'else 0) and c['tile_predicates']==408 and c['shadow_pass']
 counts={k:0 for k in LIMITS}
 for r in c['rows']:
  counts[r['site']]+=1;assert r['index']==counts[r['site']] and r['mode']==mode
  assert r['input_owner_preserved'] and r['metadata_preserved'] and r['same_stream'] and r['tile']is True
  assert r['native'] is (True if mode=='shadow' else None)
  assert r['constant']==(0 if r['site']=='gdn.initial'else 1)
  assert r['physical_elements']==(16777216 if r['site']=='gdn.initial'else 1)
 assert counts==LIMITS and len(c['rows'])==408
 return True
def dispatch(d):
 r=base_dispatch(d);eq_gate(d['eq408'],d['eq408']['mode']);return r

def qualification(root):
 from eq_all_synthetic_qualification_v1.run import result_gate
 refs=json.loads((Path(root)/'PREREQUISITE.json').read_text())
 for tag in ('base','synthetic'):
  ds={}
  for k,v in refs[tag].items():
   assert sha(v['path'])==v['sha256'];ds[k]=json.loads(Path(v['path']).read_text())
  if tag=='base':assert base_audit(ds['result'],Path(refs[tag]['result']['path']).parent.parent)['passed']
  else:assert result_gate(ds['result'])['raw_pass']
  assert ds['launch']['returncode']==ds['recovery']['actual_returncode']==0 and ds['recovery']['child_absent'] and not ds['recovery']['remaining_process_group'] and ds['recovery']['lock_available']
  assert ds['launch']['result_sha256']==ds['recovery']['result_sha256']==refs[tag]['result']['sha256']
 return dict(passed=True,evidence=refs)
def audit(r,root=None):
 try:
  assert structural_audit(r,root)['passed']
  eq_gate(r['native_request']['selected_dispatch']['eq408'],'shadow');eq_gate(r['lean_request']['selected_dispatch']['eq408'],'replace')
  return dict(passed=True,scope='EQ408 exact bool/read-only/endpoint; finite472 unchanged')
 except (AssertionError,KeyError,TypeError,ValueError)as e:return dict(passed=False,errors=[repr(e)])

def structural_audit(r,root=None):
 try:
  root=Path(root or Path(__file__).resolve().parent)
  assert r['complete']and r['candidate_checks_passed']and r['phase']=='complete'and not r.get('error')and not r.get('cleanup_errors')and r['gpu_executed']
  assert r['scope']=='current-eager lean eq408 shadow-to-replace correctness only'and r['timed_requests']==0
  assert r['engine_options']['compilation_config']==dict(mode=0,cudagraph_mode='NONE')and r['engine_options']['enforce_eager']
  c=r['cache_preseed'];assert c['destination']==str(root/'cache_v1')and c['files_verified']and c['separate_inodes_verified']
  assert r['precision_policy']['initial']==dict(bf16=True,split_k=True)and r['precision_policy']['restored']and r['precision_policy']['restored_split_k']
  assert r['adapter_install'][0]['install_order']==list(ORDER)
  native,lean=r['native_request'],r['lean_request'];assert native['mode']=='native'and native['selected_dispatch']is not None and lean['mode']=='lean'
  ft,fp,opts=frozen_reference();assert r['engine_options']==opts
  for arm in(native,lean):
   exact_tokens(arm['output_token_ids'],ft);exact_public(arm['public_logprobs'],fp)
   assert arm['allow_bf16_reduced_precision_reduction']is True and arm['allow_bf16_reduced_precision_reduction_split_k']is True
  exact_tokens(native['output_token_ids'],lean['output_token_ids']);exact_public(native['public_logprobs'],lean['public_logprobs'])
  dispatch(native['selected_dispatch']);census=dispatch(lean['selected_dispatch'])
  close=r['adapter_restore'];assert len(close)==1 and close[0]['restored']and close[0]['close_order']==list(reversed(ORDER))and all(x['restored']for x in close[0]['components'].values())
  assert r['synchronized_before_close']and r['synchronized_after_close']and r['adapter_released']and r['engine_shutdown']
  assert r['adapter_release_absent']==[True]
  return dict(passed=True,dispatch=census,scope='End-to-end512 IDs/public2560exact + lean census/restore; old per-intermediate raw proof remains separate')
 except (AssertionError,KeyError,TypeError,IndexError,ValueError)as e:return dict(passed=False,errors=[repr(e)])
