def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['public_logprob_gate']['passed'] and r['adapter_restore'][0]['restored']
  assert r['precision_policy']['restored'] and r['precision_policy']['restored_split_k'] and r['precision_policy']['policy']=='unchanged native'
  assert r['public_logprob_gate']['exact_numeric'] and r['public_logprob_gate']['values']==2560 and not r.get('cleanup_errors')
  a,b=r['shadow_request'],r['replacement_request']
  for arm,native in ((a,24),(b,0)):
   d=arm['selected_dispatch'];assert d['shadow_pass'] and d['supported_calls']==d['tileir_gate_calls']==24 and d['native_gate_calls']==native and d['supported_vendor_fallback_calls']==0
   assert len(d['rows'])==24 and len({x['path'] for x in d['rows']})==24 and all(x['passed'] and x['phase']=='prefill' and x['initial_state_all_zero'] and x['input_preserved'] for x in d['rows'])
   assert arm['cross_runtime_tokens_equal'] is True
  assert a['output_token_ids']==b['output_token_ids'] and len(a['output_token_ids'])==32 and all(len(t)==16 for t in a['output_token_ids'])
  for k in ('allow_bf16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction_split_k'):assert a[k] is True and b[k] is True
  assert not r.get('error') and not r.get('cycles') and not r.get('warmup_records')
  return dict(passed=True,errors=[],scope='24 fresh zero-initial prefill cores only; no speed or whole-library coverage claim')
 except (AssertionError,KeyError,TypeError):return dict(passed=False,errors=['Incomplete GDN prefill24 shadow/replacement correctness'])
