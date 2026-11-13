def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['adapter_restore'][0]['restored']
  assert r['precision_policy']['restored'] and r['precision_policy']['restored_split_k'] and r['precision_policy']['policy']=='unchanged native'
  for key,n in (('shadow_request',16),('replacement_request',0)):
   a=r[key];d=a['selected_dispatch'];assert a['cross_runtime_tokens_equal'] is True and a['generated_tokens']==512
   assert d['shadow_pass'] and d['supported_calls']==16 and d['native_gather_calls']==n and d['tileir_gather_calls']==16 and d['supported_vendor_fallback_calls']==0
   assert [x['call'] for x in d['rows']]==list(range(1,17))
   assert all(x['passed'] and x['finite'] and x['layout'] and x['fresh'] and x['input_unchanged'] and x['post_method_inputs_preserved'] for x in d['rows'])
   assert all((x['raw_equal'] is True and x['strict'] is True) if n else (x['raw_equal'] is None and x['strict'] is None) for x in d['rows'])
   assert a['allow_bf16_reduced_precision_reduction'] is True and a['allow_bf16_reduced_precision_reduction_split_k'] is True
  assert r['shadow_request']['output_token_ids']==r['replacement_request']['output_token_ids']
  g=r['public_logprob_gate'];assert g['passed'] and g['values']==2560 and g['exact_numeric'] and g['max_abs']==0
  assert not r.get('error') and not r.get('cleanup_errors')
  return dict(passed=True,errors=[],scope='16 hidden gathers; no performance claim')
 except (AssertionError,KeyError,TypeError):return dict(passed=False,errors=['Hidden gather integration gate failed'])
