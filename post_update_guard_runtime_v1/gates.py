import json
from pathlib import Path
from lean26_gdn_finite_runtime_v1.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch
from lean26_gdn_finite_runtime_v1.adapter import ORDER,SPECS
def structural_audit(r,root=None):
 try:
  root=Path(root or Path(__file__).resolve().parent)
  assert r['complete']and r['candidate_checks_passed']and r['phase']=='complete'and not r.get('error')and not r.get('cleanup_errors')and r['gpu_executed']
  assert r['scope']=='current-eager post-update guard16 shadow-to-replace correctness only'and r['timed_requests']==0
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

def qualification(root):
 from post_update_guard_qualification_v1.run import result_gate
 from residual_four_lean26_runtime_v1.gates import audit as four_audit
 refs=json.loads((Path(root)/'PREREQUISITE.json').read_text())
 assert set(refs)=={'four','guard'}
 for label,items in refs.items():
  docs={}
  for k,v in items.items():assert sha(v['path'])==v['sha256'];docs[k]=json.loads(Path(v['path']).read_text())
  r,l,z=docs['result'],docs['launch'],docs['recovery']
  assert (four_audit(r,Path(items['result']['path']).parent.parent)['passed']if label=='four'else result_gate(r)['raw_pass'])
  assert l['returncode']==z['actual_returncode']==0 and z['child_absent']and not z['remaining_process_group']and z['lock_available']
  assert l['result_sha256']==z['result_sha256']==items['result']['sha256']and z['launch_sha256']==items['launch']['sha256']
 return dict(passed=True,evidence=refs,scope='Prior raw four+lean26 and isolated guard; new composed predicate proof pending')

def audit(r,root=None):
 try:
  assert structural_audit(r,root)['passed']
  assert r['post_guard_binding_restored']and not r.get('post_guard_cleanup_errors')
  assert r['uva_constructor_restored']and r['uva_release']['released']and r['uva_release']['after_synchronization']and not r.get('four_cleanup_errors')
  assert r['four_replacement']['restored']and r['four_replacement']['complete']
  for key,mode in [('native_request','shadow'),('lean_request','replace')]:
   c=r[key]['post_guard'];assert c['mode']==mode and c['calls']==16 and c['shadow_pass']and not c['failed']and len(c['rows'])==16
   for i,row in enumerate(c['rows'],1):
    assert row['mode']==mode and row['index']==i and row['native_calls']==(1 if mode=='shadow'else 0)and row['candidate_calls']==1
    assert row['passed']and row['readonly']and row['metadata_preserved']and row['same_stream']and row['native_error']is None and row['candidate_error']is None
    assert row['stages']==list(range(9))and 0<row['owner_bytes']<=64*1024*1024
   f=r[key]['four_phase'];assert f['complete']and not f['failed']and f['calls']==dict(prepare_rope_positions=16,combine_sampled_and_draft_tokens=16,bias=16,post_update=16)
   assert f['raw_rechecked']is False and f['cpu_oracle_calls']==0
   assert all(row['passed']and row['original_addresses_preserved']and row['preflight_passed']and row['return_contract']for row in f['rows'])
  return dict(passed=True,guards_per_phase=16,stages_per_valid_guard=9,scope='Original typed backing predicate readonly and endpoint; no new numerical raw/physical coverage/performance proof')
 except(AssertionError,KeyError,TypeError,ValueError)as e:return dict(passed=False,errors=[repr(e)])
