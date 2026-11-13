def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['public_logprob_gate']['passed'] and r['adapter_restore'][0]['restored']
  assert r['precision_policy']['restored'] and r['precision_policy']['restored_split_k'] and r['precision_policy']['policy']=='unchanged native'
  a,b=r['shadow_request'],r['replacement_request']
  assert a['selected_dispatch']['shadow_pass'] and b['selected_dispatch']['shadow_pass']
  for arm,native in ((a,16),(b,0)):
   d=arm['selected_dispatch'];assert d['supported_calls']==d['tileir_gate_calls']==16 and d['native_gate_calls']==native and d['supported_vendor_fallback_calls']==0
   assert len(d['rows'])==16 and all(x['passed'] for x in d['rows']) and sum(x['phase']=='prefill' for x in d['rows'])==1
   assert arm['cross_runtime_tokens_equal'] is True
  assert a['output_token_ids']==b['output_token_ids'] and len(a['output_token_ids'])==32 and all(len(t)==16 for t in a['output_token_ids'])
  for k in ('allow_bf16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction_split_k'):assert a[k]==b[k]
  assert r['public_logprob_gate']['values']==2560 and r['public_logprob_gate']['exact_numeric'] and r['public_logprob_gate']['max_abs']==0
  assert a['allow_bf16_reduced_precision_reduction'] is True and a['allow_bf16_reduced_precision_reduction_split_k'] is True
  assert not r.get('cleanup_errors')
  assert not r.get('error') and not r.get('cycles') and not r.get('warmup_records')
  return dict(passed=True,errors=[],scope='greedy sampled-token full16 functional integration; no performance or ABI claim')
 except (AssertionError,KeyError,TypeError):return dict(passed=False,errors=['Incomplete greedy sampled-token shadow/replacement correctness'])
