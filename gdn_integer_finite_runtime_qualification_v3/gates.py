"""CPU result/evidence gates; no numerical kernel implementation."""
import hashlib,json,math
from pathlib import Path
from gdn_integer_finite_runtime_qualification_v3.adapter import ORDER,SPECS
from all_gemm_runtime_integration_v1.contract import plan,phase_shape
P=Path(__file__).resolve().parent.parent
OLD=P/'all_gemm_sixteen_sampler_composition_v2'
OLD_RESULT_SHA='c45d43017d5596ef12aa2e58ca1d8c29374bbfc650ca0380aaac55bd74051636'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def qualification(root):
 from lean26_gdn_finite_runtime_v1.gates import audit as prior
 q=json.loads((Path(root)/'PREREQUISITE.json').read_text());e={}
 for k,v in q.items():
  assert sha(v['path'])==v['sha256'];e[k]=json.loads(Path(v['path']).read_text())
 assert prior(e['result'])['passed']
 l,z=e['launch'],e['recovery'];assert l['returncode']==z['actual_returncode']==0 and l['child_absent'] and z['child_absent'] and not l['remaining_process_group'] and not z['remaining_process_group'] and l['lock_reacquired'] and z['lock_available']
 assert l['result_sha256']==z['result_sha256']==q['result']['sha256']
 for p,h in json.loads((Path(root)/'PINS.json').read_text()).items():assert sha(p)==h,p
 return dict(prior_actual_exit=True)
def frozen_reference():
 p=OLD/'allGemmSixteenSampler02/RESULT.json';assert sha(p)==OLD_RESULT_SHA
 r=json.loads(p.read_text());return r['replacement_request']['output_token_ids'],r['replacement_request']['public_logprobs'],r['engine_options']
def exact_tokens(x,y):
 assert len(x)==len(y)==32 and all(len(a)==len(b)==16 for a,b in zip(x,y))
 assert all(type(t)is int for row in x for t in row)and x==y,'512 generated token IDs must match exactly'
 return dict(exact=True,values=512)
def exact_public(a,b):
 assert len(a)==len(b)==32;count=0
 for ra,rb in zip(a,b):
  assert len(ra)==len(rb)==16
  for sa,sb in zip(ra,rb):
   assert len(sa)==len(sb)==5 and set(sa)==set(sb),'Exactly five matching public token keys required'
   for key in sa:
    av,bv=sa[key]['logprob'],sb[key]['logprob'];assert math.isfinite(av)and math.isfinite(bv)
    assert av==bv and sa[key]['rank']==sb[key]['rank'],'Exact public logprob/rank mismatch'
    count+=1
 assert count==2560
 return dict(exact_numeric=True,values=count,tolerance=None,scope='Public returned numeric values/keys/ranks; not full-vocabulary logits or intermediate bit patterns')
def dispatch(d):
 assert d['selected_boundaries']==5889 and d['selected_logical_operations']==6057 and d['failed']is None
 assert d['selected_native_calls']==d['selected_fallback_calls']==0
 parts=d['components'];assert set(parts)==set(ORDER)
 for key,cls,n in SPECS:
  x=parts[key];assert x['calls']==n and x['selected_native_calls']==x['selected_fallback_calls']==0
  assert sum(v['calls']for v in x['owners'])==n and len({(v['owner'],v['phase'])for v in x['owners']})==len(x['owners'])
 g=parts['all_gemm'];assert g['k128_calls']==360 and g['existing_calls']==2088
 p=plan();expected={(x['path'],phase):n for x in p['selected']for phase,n in [('prefill',1),('decode',15)]}
 assert {(r['owner'],r['phase']):r['calls']for r in g['owners']}==expected
 contracts={}
 for spec in p['selected']:
  for i in range(16):
   phase,shape=phase_shape(spec,i);key='/'.join(map(str,shape));contracts[key]=contracts.get(key,0)+1
 assert len(contracts)==13 and sum(contracts.values())==2448
 assert parts['prepare_prefill_inputs']['calls']==1
 assert parts['indexed_decode_conv']['delegated_prefill_tileir']==24
 from gdn_integer_finite_runtime_qualification_v3.core import FLOAT,INTEGER,ORDER as GDN_ORDER
 gf=parts['gdn_prefill_core']['gdn_finite'];assert gf['float_site_counts']==dict.fromkeys(FLOAT,24)and gf['typed_integer_site_counts']==dict.fromkeys(INTEGER,24)
 assert gf['float_checks']==360 and gf['typed_integer_checks']==72 and gf['host_scalar_status_retained']is True and set(gf['actual_descriptors'])==set(GDN_ORDER)
 assert gf['integer_gpu_kernels']==0 and gf['integer_stream_synchronizations']==72 and gf['integer_native_predicates']==0
 ip=parts['gdn_prefill_core']['gdn_integer_proof'];assert ip['calls']==72 and ip['native_diagnostic_predicates']==72 and ip['candidate_gpu_kernels']==0 and ip['candidate_stream_syncs']==72 and ip['raw_readonly'] is True
 assert ip['phase'] in ('shadow','lean') and ip['selected_native_predicates']==(72 if ip['phase']=='shadow' else 0) and ip['native_oracle_predicates']==72
 six={}
 for key in ('hidden_gather','sampler_fp32','sampler_four'):
  for site,record in parts[key]['finite_sites'].items():
   assert site not in six;six[site]=record
 from lean_finite_guard_runtime_v4.bindings import SITES
 assert set(six)==set(SITES)
 for record in six.values():
  assert record==dict(attempts=16,passes=16,original_selected_predicate_calls=0,host_status_item_retained=True)
 assert sum(x['passes']for x in six.values())==96
 s=parts['sampler_four'];f=s['finite_guard'];assert f['attempts']==f['passes']==f['top5_api_returns']==16 and f['original_torch_finite_calls']==0 and f['host_scalar_status_retained']is True
 assert s['logical_suboperation_calls']==64 and s['suboperation_counts']==dict(top5=16,ranks=16,selected_logprobs=16,cat=16)and not s['native_cat_retained']
 assert parts['runner_padding']['logical_suboperation_calls']==64 and parts['gdn_parent_state']['logical_suboperation_calls']==96
 return dict(passed=True,boundaries=5889,logical_operations=6057,gemm_contract_calls=contracts,gemm_contract_scope='Derived from fixed153owner/phase descriptor guards and census, not new per-call tensor hashes')
def audit(r,root=None):
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
  edge=r['finite_edge_probe'];assert edge['passed'] and len(edge['rows'])==12
  assert [(x['dtype'],x['n'],x['label'])for x in edge['rows']]==[(dtype,n,label)for dtype,n in [('torch.float32',7946240),('torch.bfloat16',81920),('torch.bfloat16',32)]for label in ('finite','nan_first','posinf_last','neginf_tail')]
  assert all(x['input_raw_preserved']and type(x['reference'])is bool and x['reference']==(x['label']=='finite')and x['repeats']==[x['reference']]*2 for x in edge['rows'])
  shadow=r['shadow_request'];assert shadow['mode']=='shadow'
  exact_tokens(shadow['output_token_ids'],ft);exact_public(shadow['public_logprobs'],fp)
  assert shadow['allow_bf16_reduced_precision_reduction'] is True and shadow['allow_bf16_reduced_precision_reduction_split_k'] is True
  dispatch(shadow['selected_dispatch']);assert shadow['selected_dispatch']['components']['gdn_prefill_core']['gdn_integer_proof']['phase']=='shadow'
  census=dispatch(lean['selected_dispatch']);assert lean['selected_dispatch']['components']['gdn_prefill_core']['gdn_integer_proof']['phase']=='lean'
  close=r['adapter_restore'];assert len(close)==1 and close[0]['restored']and close[0]['close_order']==list(reversed(ORDER))and all(x['restored']for x in close[0]['components'].values())
  assert r['synchronized_before_close']and r['synchronized_after_close']and r['adapter_released']and r['engine_shutdown']
  assert r['adapter_release_absent']==[True]
  return dict(passed=True,dispatch=census,scope='End-to-end512 IDs/public2560exact + lean census/restore; old per-intermediate raw proof remains separate')
 except (AssertionError,KeyError,TypeError,IndexError,ValueError)as e:return dict(passed=False,errors=[repr(e)])
