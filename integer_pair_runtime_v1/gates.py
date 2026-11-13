from pathlib import Path
import json
from lean26_gdn_finite_runtime_v1.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch as lean_dispatch
from lean26_gdn_finite_runtime_v1.adapter import ORDER,SPECS

def pair_gate(pair,mode):
 for name,n in [('embedding_cast',16),('sampler_integer',48)]:
  c=pair[name];assert c['mode']==mode and c['calls']==n and len(c['rows'])==n
  assert c['native_calls']==(n if mode=='shadow'else 0) and c['tile_calls']==(n if mode=='replace'else 0)
  for i,row in enumerate(c['rows']):
   assert row['index']==i and row['mode']==mode and row['passed'] and row['fresh'] and row['metadata_preserved'] and row['raw_compared']is False
   if name=='embedding_cast':assert row['n']==(1696 if i==0 else 32)
   else:assert row['site']==('positions','input_ids','ones')[i%3]
 assert pair['sampler_integer']['site_counts']==dict(positions=16,input_ids=16,ones=16)
 return True

def qualification(root):
 from embedding_int64_runtime_v2.gates import audit as cast_audit
 from sampler_integer_runtime_v1.gates import audit as integer_audit
 refs=json.loads((Path(root)/'PREREQUISITE.json').read_text())
 for tag,auditor in [('cast',cast_audit),('integer',integer_audit)]:
  d={}
  for k,v in refs[tag].items():
   assert sha(v['path'])==v['sha256'];d[k]=json.loads(Path(v['path']).read_text())
  assert auditor(d['result'],Path(refs[tag]['result']['path']).parent.parent)['passed']
  assert d['launch']['returncode']==d['recovery']['actual_returncode']==0
  assert d['recovery']['child_absent'] and not d['recovery']['remaining_process_group'] and d['recovery']['lock_available']
  assert d['launch']['result_sha256']==d['recovery']['result_sha256']==refs[tag]['result']['sha256']
 return dict(passed=True,evidence=refs)

def dispatch(d):
 r=lean_dispatch(d);pair_gate(d['integer_pair'],'replace');return r

def audit(r,root=None):
 try:
  assert structural_audit(r,root)['passed'];pair_gate(r['native_request']['integer_pair_shadow'],'shadow')
  assert all(r['adapter_restore'][0]['components'][k]['restored'] for k in ('lean','sampler_integer','embedding_cast'))
  return dict(passed=True,scope='64 integer endpoint calls, prior raw proofs separate; no performance claim')
 except (AssertionError,KeyError,TypeError,ValueError)as e:return dict(passed=False,errors=[repr(e)])

def structural_audit(r,root=None):
 try:
  root=Path(root or Path(__file__).resolve().parent)
  assert r['complete']and r['candidate_checks_passed']and r['phase']=='complete'and not r.get('error')and not r.get('cleanup_errors')and r['gpu_executed']
  assert r['scope']=='current-eager native-to-lean correctness only'and r['timed_requests']==0
  assert r['engine_options']['compilation_config']==dict(mode=0,cudagraph_mode='NONE')and r['engine_options']['enforce_eager']
  c=r['cache_preseed'];assert c['destination']==str(root/'cache_v1')and c['files_verified']and c['separate_inodes_verified']
  assert r['precision_policy']['initial']==dict(bf16=True,split_k=True)and r['precision_policy']['restored']and r['precision_policy']['restored_split_k']
  assert r['adapter_install'][0]['install_order']==list(ORDER)
  native,lean=r['native_request'],r['lean_request'];assert native['mode']=='native'and native['selected_dispatch']is None and lean['mode']=='lean'
  ft,fp,opts=frozen_reference();assert r['engine_options']==opts
  for arm in(native,lean):
   exact_tokens(arm['output_token_ids'],ft);exact_public(arm['public_logprobs'],fp)
   assert arm['allow_bf16_reduced_precision_reduction']is True and arm['allow_bf16_reduced_precision_reduction_split_k']is True
  exact_tokens(native['output_token_ids'],lean['output_token_ids']);exact_public(native['public_logprobs'],lean['public_logprobs'])
  census=dispatch(lean['selected_dispatch'])
  close=r['adapter_restore'];assert len(close)==1 and close[0]['restored']and close[0]['components']['lean']['close_order']==list(reversed(ORDER))and all(x['restored']for x in close[0]['components'].values())
  assert r['synchronized_before_close']and r['synchronized_after_close']and r['adapter_released']and r['engine_shutdown']
  assert r['adapter_release_absent']==[True]
  return dict(passed=True,dispatch=census,scope='End-to-end512 IDs/public2560exact + lean census/restore; old per-intermediate raw proof remains separate')
 except (AssertionError,KeyError,TypeError,IndexError,ValueError)as e:return dict(passed=False,errors=[repr(e)])
