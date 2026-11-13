def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['adapter_restore'][0]['restored'] and not r.get('error') and not r.get('cleanup_errors')
  assert r['precision_policy']['restored'] and r['precision_policy']['restored_split_k'] and r['precision_policy']['policy']=='unchanged native'
  for key,n in [('shadow_request',16),('replacement_request',0)]:
   a=r[key];d=a['selected_dispatch'];assert d['shadow_pass'] and d['supported_calls']==16 and d['native_scatter_calls']==n and d['tile_scatter_calls']==16 and d['fallback_calls']==0
   assert [x['call']for x in d['rows']]==list(range(1,17)) and all(all(x[k]for k in ('passed','raw_equal','full_owner_equal','input_preserved','layout','return_none'))for x in d['rows'])
   assert a['cross_runtime_tokens_equal']is True and a['generated_tokens']==512 and a['allow_bf16_reduced_precision_reduction']is True and a['allow_bf16_reduced_precision_reduction_split_k']is True
  g=r['public_logprob_gate'];assert g['passed'] and g['values']==2560 and g['exact_numeric'] and g['max_abs']==0
  assert r['shadow_request']['output_token_ids']==r['replacement_request']['output_token_ids']
  return dict(passed=True,errors=[],scope='Accepted scatter16 full-model functional integration; no performance claim')
 except (AssertionError,KeyError,TypeError):return dict(passed=False,errors=['Accepted scatter integration incomplete'])
