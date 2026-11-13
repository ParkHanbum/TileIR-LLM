"""CPU result/evidence gates; no numerical kernel implementation."""
import hashlib,json,math
from pathlib import Path
from lean26_gdn_finite_runtime_v1.adapter import ORDER,SPECS
from all_gemm_runtime_integration_v1.contract import plan,phase_shape
P=Path(__file__).resolve().parent.parent
OLD=P/'all_gemm_sixteen_sampler_composition_v2'
OLD_RESULT_SHA='c45d43017d5596ef12aa2e58ca1d8c29374bbfc650ca0380aaac55bd74051636'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def qualification(root):
 from lean26_prefill_qualification_v1.gates import audit as lean26_audit
 from lean_finite_guard_runtime_v4.gates import audit as finite_audit
 from gdn_large_finite_qualification_v1.run import result_gate
 refs=json.loads((Path(root)/'PREREQUISITE.json').read_text())
 for key,ref in refs.items():
  data={}
  for kind,item in ref.items():
   assert sha(item['path'])==item['sha256'];data[kind]=json.loads(Path(item['path']).read_text())
  rr,ll,zz=data['result'],data['launch'],data['recovery']
  passed=result_gate(rr)['raw_pass']if key=='large'else (lean26_audit(rr,Path(ref['result']['path']).parent.parent)if key=='lean26'else finite_audit(rr,Path(ref['result']['path']).parent.parent))['passed']
  assert passed and ll['returncode']==zz['actual_returncode']==0
  assert zz['child_absent']and not zz['remaining_process_group']and zz.get('lock_available',zz.get('lock_reacquired',False))
  assert ll['result_sha256']==zz['result_sha256']==ref['result']['sha256']and zz['launch_sha256']==ref['launch']['sha256']
 assert set(refs)=={'lean26','finite112','large'}
 return dict(passed=True,evidence=refs,scope='Three prior qualifications; combined GDN360 model remains pending')
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
 from lean26_gdn_finite_runtime_v1.core import FLOAT,INTEGER,ORDER as GDN_ORDER
 gf=parts['gdn_prefill_core']['gdn_finite'];assert gf['float_site_counts']==dict.fromkeys(FLOAT,24)and gf['retained_integer_site_counts']==dict.fromkeys(INTEGER,24)
 assert gf['float_checks']==360 and gf['retained_integer_checks']==72 and gf['host_scalar_status_retained']is True and set(gf['actual_descriptors'])==set(GDN_ORDER)
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
  census=dispatch(lean['selected_dispatch'])
  close=r['adapter_restore'];assert len(close)==1 and close[0]['restored']and close[0]['close_order']==list(reversed(ORDER))and all(x['restored']for x in close[0]['components'].values())
  assert r['synchronized_before_close']and r['synchronized_after_close']and r['adapter_released']and r['engine_shutdown']
  assert r['adapter_release_absent']==[True]
  return dict(passed=True,dispatch=census,scope='End-to-end512 IDs/public2560exact + lean census/restore; old per-intermediate raw proof remains separate')
 except (AssertionError,KeyError,TypeError,IndexError,ValueError)as e:return dict(passed=False,errors=[repr(e)])
