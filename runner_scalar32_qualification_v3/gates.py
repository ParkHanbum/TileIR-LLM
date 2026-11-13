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
 evidence={}
 for k,v in q['evidence'].items():
  assert sha(v['path'])==v['sha256'];evidence[k]=json.loads(Path(v['path']).read_text())
 assert evidence['result']['complete'] is True
 l,z=evidence['launch'],evidence['recovery']
 assert l['returncode']==z['actual_returncode']==0 and l['child_absent'] and z['child_absent']
 assert not l['remaining_process_group'] and not z['remaining_process_group'] and l['lock_reacquired'] and z['lock_available']
 assert l['result_sha256']==z['result_sha256']==q['evidence']['result']['sha256']==OLD_RESULT_SHA
 return {'source_verified':True,'frozen_endpoint_actual_exit_verified':True}
def audit(r,root=None):
 try:
  root=Path(root or Path(__file__).parent)
  assert r['complete'] is True and r['candidate_checks_passed'] is True and not r.get('error') and not r.get('cleanup_errors')
  ft,fp,options=frozen_reference();assert r['engine_options']==options
  for mode in ('native','shadow','replace'):
   x=r[mode+'_request'];exact_tokens(x['output_token_ids'],ft);exact_public(x['public_logprobs'],fp)
   assert x['allow_bf16_reduced_precision_reduction'] is True and x['allow_bf16_reduced_precision_reduction_split_k'] is True
  for mode in ('shadow','replace'):
   c=r[mode+'_census'];assert c==dict(calls=32,indices=list(range(32)),raw_backing_bytes_per_call=128,mode=mode)
  assert r['synthetic']==dict(cases=32,passed=True)
  ledger=r['call_ledger']
  assert [x['sequence']for x in ledger]==list(range(len(ledger)))
  for phase in ('shadow','replace'):
   rows=[x for x in ledger if x['phase']==phase]
   assert not any(x['stage'] in ('failure','entry_failure')for x in rows)
   for stage in ('preimage','candidate_intent','candidate_return','verified'):
    events=[x for x in rows if x['stage']==stage];assert len(events)==32 and sorted(x['index']for x in events)==list(range(32))
   native='native_original' if phase=='shadow' else 'native_private'
   for call in range(32):
    events=[x for x in rows if x['call']==call]
    assert [x['stage']for x in events]==['preimage',native+'_intent',native+'_return','candidate_intent','candidate_return','verified']
    assert len({x['index']for x in events})==1
    before=bytes.fromhex(events[0]['before_hex']);expected=bytearray(before);assert len(before)==128
    index=events[0]['index'];expected[index*4:index*4+4]=(1).to_bytes(4,'little',signed=True)
    assert bytes(expected).hex()==events[0]['expected_hex']
    assert events[-1]['after_sha256']==hashlib.sha256(expected).hexdigest()
    assert events[3]['target']==('original' if phase=='replace' else 'private')
   assert sum(x['stage']==native+'_intent'for x in rows)==sum(x['stage']==native+'_return'for x in rows)==32
   if phase=='replace':assert not any(x['stage'].startswith('native_original')for x in rows)
   for x in rows:
    if x['stage']=='verified':assert x['fullbacking_raw'] is True and x['inactive_bytes_exact'] is True

  assert r['observer_restored'] and r['engine_shutdown'] and r['synchronized_after_restore']
  assert r['precision_policy']['initial']==dict(bf16=True,split_k=True) and r['precision_policy']['restored'] and r['precision_policy']['restored_split_k']
  assert r['owner_binding_verified'] and r['cache_preseed']['destination']==str(root/'cache_v1') and r['cache_preseed']['files_verified'] and r['cache_preseed']['separate_inodes_verified']
  return dict(passed=True,scope='scalar32 correctness only')
 except (AssertionError,KeyError,TypeError,ValueError) as e:return dict(passed=False,error=type(e).__name__)
