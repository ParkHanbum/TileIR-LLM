import hashlib,json,math
from pathlib import Path
P=Path(__file__).resolve().parent.parent
OLD=P/'all_gemm_sixteen_sampler_composition_v2'
OLD_RESULT_SHA='c45d43017d5596ef12aa2e58ca1d8c29374bbfc650ca0380aaac55bd74051636'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
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
def qualification(root):
 root=Path(root)
 for p,h in json.loads((root/'PINS.json').read_text()).items():assert sha(p)==h,p
 q=json.loads((root/'PREREQUISITE.json').read_text());assert q['complete'] is True
 e={}
 for k,v in q['evidence'].items():
  assert sha(v['path'])==v['sha256'];e[k]=json.loads(Path(v['path']).read_text())
 from runtime_metadata_five_v1.gates import audit as prior_audit
 assert prior_audit(e['result'])['passed']
 l,z=e['launch'],e['recovery']
 assert l['returncode']==z['actual_returncode']==0 and l['child_absent'] and z['child_absent']
 assert not l['remaining_process_group'] and not z['remaining_process_group'] and l['lock_reacquired'] and z['lock_available']
 assert l['result_sha256']==z['result_sha256']==q['evidence']['result']['sha256']
 return dict(metadata_five_actual_exit_verified=True,source_verified=True)
def audit(r,root=None):
 try:
  root=Path(root or Path(__file__).parent)
  assert r['complete'] is True and r['candidate_checks_passed'] is True and not r.get('error') and not r.get('cleanup_errors')
  ft,fp,options=frozen_reference();assert r['engine_options']==options
  for mode in ('native','shadow','replace'):
   x=r[mode+'_request'];exact_tokens(x['output_token_ids'],ft);exact_public(x['public_logprobs'],fp)
   assert x['allow_bf16_reduced_precision_reduction'] is True and x['allow_bf16_reduced_precision_reduction_split_k'] is True
  for mode in ('shadow','replace'):
   c=r[mode+'_census'];assert c['mode']==mode and c['calls']==1 and len(c['rows'])==1
   row=c['rows'][0]
   assert row['mode']==mode and all(row[k] is True for k in ('raw','source_readonly','outside_preserved'))
   assert row['native_original_calls']==(1 if mode=='shadow' else 0) and row['native_private_oracles']==(0 if mode=='shadow' else 1) and row['candidate_calls']==1
   t=row['transport'];assert all(t[k]==1 for k in ('transfer_attempts','transfer_returns','cast_launch_attempts','cast_launch_returns'))
   assert t['temporary_bytes']==1792 and t['source_pinned'] is True and t['same_dtype_transfer'] is True
  assert r['synthetic']==dict(patterns=8,rows=224,phases=2,complete=True)
  assert r['observer_restored'] and r['transport_drained'] and r['engine_shutdown'] and r['synchronized_after_restore']
  assert r['metadata_progress']['pending_owners']==0 and not r['metadata_progress']['failed']
  assert r['precision_policy']['initial']==dict(bf16=True,split_k=True) and r['precision_policy']['restored'] and r['precision_policy']['restored_split_k']
  assert r['cache_preseed']['destination']==str(root/'cache_v1') and r['cache_preseed']['files_verified'] and r['cache_preseed']['separate_inodes_verified']
  return dict(passed=True,scope='Single cast; replacement private native oracle and native H2D retained')
 except (AssertionError,KeyError,TypeError,ValueError) as e:return dict(passed=False,error=type(e).__name__)
