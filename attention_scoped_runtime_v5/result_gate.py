from .adapter import census
def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['adapter_restore'][0]['restored'] and not r.get('error') and not r.get('cleanup_errors')
  p=r['precision_policy'];assert p['policy']=='unchanged native' and p['initial']==dict(bf16=True,split_k=True) and p['restored'] and p['restored_split_k']
  a,b=r['shadow_request'],r['replacement_request']
  for arm,native in ((a,128),(b,0)):
   d=arm['selected_dispatch'];assert d['shadow_pass'] and not d['failed'];census(d['rows'])
   assert d['total_calls']==128 and d['supported_calls']==d['tileir_gate_calls']==128 and d['native_gate_calls']==native and d['explicit_native_exceptions']==0 and d['supported_vendor_fallback_calls']==0
   assert sum(x['native_calls'] for x in d['rows'])==native and sum(x['tile_calls'] for x in d['rows'])==128
   assert all(x['native_calls']==(1 if arm is a or not x['supported'] else 0) and x['tile_calls']==int(x['supported']) for x in d['rows'])
   assert {s['length'] for s in d['late_banks']}=={65,68} and len(d['late_banks'])==2
   assert arm['cross_runtime_tokens_equal'] is True and arm['allow_bf16_reduced_precision_reduction'] is True and arm['allow_bf16_reduced_precision_reduction_split_k'] is True
  assert a['output_token_ids']==b['output_token_ids'] and len(a['output_token_ids'])==32 and all(len(x)==16 for x in a['output_token_ids'])
  g=r['public_logprob_gate'];assert g['passed'] and g['values']==2560 and g['exact_numeric'] and g['max_abs']==0
  return dict(passed=True,scope='128 actual attention calls replaced; no whole-owner/performance/other-workload claim')
 except (AssertionError,KeyError,TypeError):return dict(passed=False,errors=['Incomplete scoped attention correctness'])
