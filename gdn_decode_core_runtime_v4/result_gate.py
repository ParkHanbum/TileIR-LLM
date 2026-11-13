def audit(r):
 try:
  assert r['complete']and r['engine_shutdown']and r['adapter_restore'][0]['restored']
  g=r['public_logprob_gate'];assert g['passed']and g['exact_numeric']and g['max_abs']==0 and g['values']==2560
  p=r['precision_policy'];assert p['restored']and p['restored_split_k']and p['changes']==[]
  a,b=r['shadow_request'],r['replacement_request'];assert a['output_token_ids']==b['output_token_ids']and len(a['output_token_ids'])==32 and all(len(x)==16 for x in a['output_token_ids'])
  for arm,original in [(a,360),(b,0)]:
   assert arm['cross_runtime_tokens_equal']is True and arm['generated_tokens']==512
   d=arm['selected_dispatch'];assert d['shadow_pass']and d['supported_calls']==360 and d['supported_vendor_fallback_calls']==0 and d['other_bindings_unchanged']
   assert len(d['counts'])==24 and set(d['counts'].values())=={15}and len(d['rows'])==360 and all(v['passed']for v in d['rows'])
   assert sum(x['original_calls']for x in d['rows'])==original and sum(x['tileir_calls']for x in d['rows'])==360
  assert not r.get('error')and not r.get('cycles');return dict(passed=True,errors=[])
 except(AssertionError,KeyError,TypeError):return dict(passed=False,errors=['GDN ordinary decode full functional HOLD'])
