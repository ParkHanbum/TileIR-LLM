from pathlib import Path
import json
from lean26_gdn_finite_runtime_v1.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch as lean_dispatch,audit as prior_audit
from lean26_gdn_finite_runtime_v1.adapter import ORDER,SPECS
def cast_gate(c,mode):
 assert c['mode']==mode and c['calls']==c['tile_cast_calls']==16 and c['native_cast_calls']==(16 if mode=='shadow'else 0)and c['shadow_pass']
 assert len(c['rows'])==16
 for i,r in enumerate(c['rows']):
  assert r['index']==i and r['n']==(1696 if i==0 else 32)and r['mode']==mode and r['passed']and r['input_owner_preserved']and r['metadata_preserved']and r['fresh']
  assert r['raw_compared']==r['returned_native_identity']==(mode=='shadow')
 return True
def qualification(root):
 refs=json.loads((Path(root)/'PREREQUISITE.json').read_text());d={}
 for k,v in refs.items():assert sha(v['path'])==v['sha256'];d[k]=json.loads(Path(v['path']).read_text())
 assert prior_audit(d['result'],Path(refs['result']['path']).parent.parent)['passed']
 assert d['launch']['returncode']==d['recovery']['actual_returncode']==0 and d['recovery']['child_absent']and not d['recovery']['remaining_process_group']and d['recovery']['lock_available']
 assert d['recovery']['result_sha256']==d['launch']['result_sha256']==refs['result']['sha256']
 return dict(passed=True,evidence=refs)
def dispatch(d):
 result=lean_dispatch(d);cast_gate(d['embedding_cast'],'replace');return result
def audit(r,root=None):
 # Exact prior structural checks, with current cast edge receipt checked separately.
 try:
  assert structural_audit(r,root or Path(__file__).parent)['passed']
  cast_gate(r['native_request']['embedding_cast_shadow'],'shadow');cast_gate(r['lean_request']['selected_dispatch']['embedding_cast'],'replace')
  e=r['cast_edge_probe'];assert e['passed']and [x['n']for x in e['records']]==[32,1696]and all(x['repeats']==2 and x['raw_exact']and x['signed_exact']and x['input_preserved']and x['fresh']for x in e['records'])
  assert r['adapter_restore'][0]['embedding_cast_restore']['restored']
  return dict(passed=True,scope='Native cast shadow16 then conditional cast replacement16; endpoint plus prior lean26 gates; synthetic cast edges separate')
 except (AssertionError,KeyError,TypeError)as e:return dict(passed=False,errors=[repr(e)])

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
  close=r['adapter_restore'];assert len(close)==1 and close[0]['restored']and close[0]['close_order']==list(reversed(ORDER))and all(x['restored']for x in close[0]['components'].values())
  assert r['synchronized_before_close']and r['synchronized_after_close']and r['adapter_released']and r['engine_shutdown']
  assert r['adapter_release_absent']==[True]
  return dict(passed=True,dispatch=census,scope='End-to-end512 IDs/public2560exact + lean census/restore; old per-intermediate raw proof remains separate')
 except (AssertionError,KeyError,TypeError,IndexError,ValueError)as e:return dict(passed=False,errors=[repr(e)])
