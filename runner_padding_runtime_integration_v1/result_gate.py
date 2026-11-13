def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['adapter_restore'][0]['restored']
  assert r['precision_policy']['restored'] and r['precision_policy']['restored_split_k'] and r['precision_policy']['policy']=='unchanged native'
  for key,n in (('shadow_request',32),('replacement_request',0)):
   a=r[key];d=a['selected_dispatch'];assert a['cross_runtime_tokens_equal'] is True and a['generated_tokens']==512
   assert d['shadow_pass'] and d['supported_calls']==16 and d['logical_suboperation_calls']==64 and d['native_constructor_calls']==n and d['tileir_constructor_calls']==32 and d['supported_vendor_fallback_calls']==0
   assert [(x['call'],x['kind']) for x in d['rows']]==[(i,k) for i in range(1,17) for k in ('cu_num_logits','expanded_local_pos')]
   assert d['native_fill_calls']==n and d['tileir_fill_calls']==32 and d['nonempty_tile_fill_calls']==16
   assert [(x['call'],x['value'])for x in d['fill_rows']]==[(i,v)for i in range(1,17)for v in (False,True)]
   assert all(x['passed'] and x['raw_equal'] and x['outside_unchanged'] and x['return_identity'] and x['layout'] for x in d['fill_rows'])
   assert sum(x['length']==1696 for x in d['fill_rows'])==1 and sum(x['length']==32 for x in d['fill_rows'])==15 and sum(x['length']==0 for x in d['fill_rows'])==16
   assert all(x['passed'] and x['raw_equal'] and x['prior_unchanged'] and x['fresh_disjoint'] for x in d['rows'])
   assert a['allow_bf16_reduced_precision_reduction'] is True and a['allow_bf16_reduced_precision_reduction_split_k'] is True
  assert r['shadow_request']['output_token_ids']==r['replacement_request']['output_token_ids']
  g=r['public_logprob_gate'];assert g['passed'] and g['values']==2560 and g['exact_numeric'] and g['max_abs']==0
  assert not r.get('error') and not r.get('cleanup_errors')
  return dict(passed=True,errors=[],scope='64 constructor/fill logical suboperations; no performance claim')
 except (AssertionError,KeyError,TypeError):return dict(passed=False,errors=['Constructor integration gate failed'])
