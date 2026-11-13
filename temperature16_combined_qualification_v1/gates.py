import json
from pathlib import Path
from lean26_gdn_finite_runtime_v1.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch as base_dispatch
from i64_range48_runtime_v1.gates import eq_gate as range_gate,qualification as range_qualification
from i64_range48_runtime_v1.adapter import ORDER
from sampling_guards144_runtime_v2.checks import LIMITS
def dispatch(d):
 r=base_dispatch(d)
 c=d['range48'];assert c['calls']==48 and c['counts']=={'greedy':16,'ranks':16,'scores':16} and c['native_predicates']==0
 c=d['sampling96'];assert c['calls']==96 and c['counts']==LIMITS and c['original_predicates']==c['original_gathers']==0 and c['temperature_followup_hook']is True
 t=d['temperature16'];assert t['calls']==16 and t['shadow_pass'] and t['mode']in('shadow','replace') and len(t['rows'])==16
 assert t['native_predicates']==(16 if t['mode']=='shadow'else 0)and c['temperature_original_retained']==t['native_predicates']
 for i,row in enumerate(t['rows']):
  assert row['index']==i+1 and row['stage']=='verified'and row['mode']==t['mode']and row['readonly']and row['same_stream']and row['tile']is True
  assert row['native']is(True if t['mode']=='shadow'else None)and row['raw_gather_compared']is(t['mode']=='shadow')
 return r

def qualification(root):
 from sampling_guards144_runtime_v2.gates import audit as parent_audit
 refs=json.loads((Path(root)/'PREREQUISITE.json').read_text());ds={}
 for k,v in refs.items():assert sha(v['path'])==v['sha256'];ds[k]=json.loads(Path(v['path']).read_text())
 assert parent_audit(ds['result'],Path(refs['result']['path']).parent.parent)['passed']
 assert ds['launch']['returncode']==ds['recovery']['actual_returncode']==0 and ds['launch']['child_absent']and ds['launch']['lock_reacquired']
 assert ds['recovery']['child_absent']and not ds['recovery']['remaining_process_group']and ds['recovery']['lock_available']
 assert ds['launch']['result_sha256']==ds['recovery']['result_sha256']==refs['result']['sha256']and ds['recovery']['launch_sha256']==refs['launch']['sha256']
 return refs

def audit(r,root=None):
 try:
  assert structural_audit(r,root)['passed']
  iso=r['isolated_temperature'];assert iso['complete']and iso['cases']==52 and iso['streams']==2 and len(iso['rows'])==52 and all(x['passed']and x['readonly']for x in iso['rows'])
  from .isolated import cases
  expected={(c['name'],stream):c.get('reject',False)for stream in range(2)for c in cases()}
  expected.update({('bf16_eq_reject',stream):True for stream in range(2)})
  actual={(x['name'],x['stream_index']):x['rejected']for x in iso['rows']}
  assert len(actual)==52 and actual==expected
  assert r['native_request']['selected_dispatch']['temperature16']['mode']=='shadow'and r['lean_request']['selected_dispatch']['temperature16']['mode']=='replace'
  return dict(passed=True,scope='Isolated52 then temperature16 actual raw/EQ0/readonly and full endpoint')
 except(AssertionError,KeyError,TypeError,ValueError)as e:return dict(passed=False,errors=[repr(e)])

def structural_audit(r,root=None):
 try:
  root=Path(root or Path(__file__).resolve().parent)
  assert r['complete']and r['candidate_checks_passed']and r['phase']=='complete'and not r.get('error')and not r.get('cleanup_errors')and r['gpu_executed']
  assert r['scope']=='current-eager temperature16 isolated-then-model correctness only'and r['timed_requests']==0
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

