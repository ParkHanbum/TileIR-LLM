import json
from pathlib import Path
from lean26_gdn_finite_runtime_v1.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch as base_dispatch
from i64_range48_runtime_v1.gates import eq_gate as range_gate,qualification as range_qualification
from i64_range48_runtime_v1.adapter import ORDER
from .checks import LIMITS
def sample_gate(c,mode):
 assert c['mode']==mode and c['calls']==96 and c['counts']==LIMITS and c['shadow_pass']
 assert c['original_predicates']==(80 if mode=='shadow'else 0)and c['original_gathers']==(16 if mode=='shadow'else 0)
 assert c['tile_predicates']==80 and c['tile_gathers']==16 and c['additional_internal_range_predicates']==16
 counts={k:0 for k in LIMITS}
 for row in c['rows']:
  counts[row['site']]+=1;assert row['index']==counts[row['site']]and row['mode']==mode and row['stage']=='verified'
  assert row['input_owner_preserved']and row['metadata_preserved']and row['same_stream']
  if row['kind']=='predicate':
   expected=row['site']=='greedy.rows';assert row['tile']is expected and row['native']is(expected if mode=='shadow'else None)
  else:
   assert row['site']=='ranks.gather'and row['raw_comparison']is(mode=='shadow')and row['output_fresh']and row['output_shape']==[32,1]and row['internal_checked_range_calls']==1
 assert counts==LIMITS and len(c['rows'])==96 and len(c['temperature_metadata'])==16
 for i,t in enumerate(c['temperature_metadata']):
  assert t['index']==i+1 and t['values_read']is False and t['original_temperature_predicate_retained']is True
  assert all(k in t['temperature']and k in t['idx']for k in ('shape','stride','dtype','storage_offset','storage_bytes','device','requires_grad','is_neg','is_conj'))
 return True
def dispatch(d):
 r=base_dispatch(d);range_gate(d['range48'],d['range48']['mode']);sample_gate(d['sampling96'],d['sampling96']['mode']);return r
def qualification(root):
 out=range_qualification(root)
 refs=json.loads((Path(root)/'PREREQUISITE.json').read_text())['sampling']
 ds={}
 for k,v in refs.items():assert sha(v['path'])==v['sha256'];ds[k]=json.loads(Path(v['path']).read_text())
 from sampling_guard_synthetic_qualification_v1.run import result_gate
 assert result_gate(ds['result'])['raw_pass']
 assert ds['launch']['returncode']==ds['recovery']['actual_returncode']==0 and ds['launch']['child_absent']and ds['launch']['lock_reacquired']
 assert ds['recovery']['child_absent']and not ds['recovery']['remaining_process_group']and ds['recovery']['lock_available']
 assert ds['launch']['result_sha256']==ds['recovery']['result_sha256']==refs['result']['sha256']
 assert ds['recovery']['launch_sha256']==refs['launch']['sha256']
 out['sampling_isolated']=refs;return out
def audit(r,root=None):
 try:
  assert structural_audit(r,root)['passed']
  for key,mode in [('native_request','shadow'),('lean_request','replace')]:
   d=r[key]['selected_dispatch'];range_gate(d['range48'],mode);sample_gate(d['sampling96'],mode)
  return dict(passed=True,scope='Joint range48 plus sampling96 actual bool/gather/raw-readonly/endpoint; temperature predicate native')
 except(AssertionError,KeyError,TypeError,ValueError)as e:return dict(passed=False,errors=[repr(e)])

def structural_audit(r,root=None):
 try:
  root=Path(root or Path(__file__).resolve().parent)
  assert r['complete']and r['candidate_checks_passed']and r['phase']=='complete'and not r.get('error')and not r.get('cleanup_errors')and r['gpu_executed']
  assert r['scope']=='current-eager lean sampling144 shadow-to-replace correctness only'and r['timed_requests']==0
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

