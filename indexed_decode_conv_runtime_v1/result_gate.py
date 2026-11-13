def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['adapter_restore'][0]['restored'] and r['public_logprob_gate']['passed']
  assert r['precision_policy']['restored'] and r['precision_policy']['restored_split_k'] and r['precision_policy']['changes']==[]
  a,b=r['shadow_request'],r['replacement_request'];assert a['output_token_ids']==b['output_token_ids'] and len(a['output_token_ids'])==32 and all(len(t)==16 for t in a['output_token_ids'])
  for arm in (a,b):
   d=arm['selected_dispatch'];assert d['shadow_pass'] and d['supported_calls']==360 and d['supported_vendor_fallback_calls']==0
   assert len(d['rows'])==360 and all(v['passed'] for v in d['rows'])
   assert sum(v for k,v in d['counts'].items() if k.endswith(':prefill'))==24 and sum(v for k,v in d['counts'].items() if k.endswith(':decode'))==360
  assert sum(x['original_calls'] for x in a['selected_dispatch']['rows'])==360 and sum(x['original_calls'] for x in b['selected_dispatch']['rows'])==0
  assert not r.get('error') and not r.get('cycles');return dict(passed=True,errors=[])
 except (AssertionError,KeyError,TypeError):return dict(passed=False,errors=['Indexed decode functional HOLD'])
