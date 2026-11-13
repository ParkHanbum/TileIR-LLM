from collections import Counter
def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['public_logprob_gate']['passed'] and r['adapter_restore'][0]['restored']
  p=r['precision_policy'];assert p['restored'] and p['restored_split_k'] and p['policy']=='unchanged native'
  assert r['public_logprob_gate']['values']==2560
  a,b=r['shadow_request'],r['replacement_request']
  for arm,native in ((a,128),(b,0)):
   d=arm['selected_dispatch'];assert d['shadow_pass'] and d['supported_calls']==d['tileir_gate_calls']==128 and d['native_gate_calls']==native and d['supported_vendor_fallback_calls']==0
   rows=d['rows'];assert len(rows)==128 and all(x['passed'] and x['v_contract'] and x['project_tuple_preserved'] and x['finite'] for x in rows)
   paths={x['path'] for x in rows};assert len(paths)==8
   assert Counter((x['path'],x['phase']) for x in rows)==Counter({(n,p):c for n in paths for p,c in [('prefill',1),('decode',15)]})
   assert arm['cross_runtime_tokens_equal'] is True
  assert a['output_token_ids']==b['output_token_ids'] and len(a['output_token_ids'])==32 and all(len(t)==16 for t in a['output_token_ids'])
  for k in ('allow_bf16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction_split_k'):assert a[k]==b[k]
  assert not r.get('cleanup_errors') and not r.get('error') and not r.get('cycles') and not r.get('warmup_records')
  return dict(passed=True,errors=[],scope='All128 eager explicit QK helper calls; no performance, graph or ABI claim')
 except (AssertionError,KeyError,TypeError):return dict(passed=False,errors=['Incomplete QK shadow/replacement correctness'])
