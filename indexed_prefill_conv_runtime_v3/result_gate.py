def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['adapter_restore'][0]['restored'] and r['adapter_restore'][0]['update_binding_is_install_time_object']
  g=r['public_logprob_gate'];assert g['passed'] and g['exact_numeric'] and g['max_abs']==0.0 and g['values']==2560
  p=r['precision_policy'];assert p['restored'] and p['restored_split_k'] and p['changes']==[] and p['initial_allow_bf16_reduced_precision_reduction']is True and p['initial_allow_bf16_reduced_precision_reduction_split_k']is True
  a,b=r['shadow_request'],r['replacement_request'];assert a['output_token_ids']==b['output_token_ids'] and len(a['output_token_ids'])==32 and all(len(t)==16 for t in a['output_token_ids']) and a['generated_tokens']==b['generated_tokens']==512
  for arm in(a,b):
   d=arm['selected_dispatch'];assert d['shadow_pass'] and d['supported_calls']==24 and d['supported_vendor_fallback_calls']==0 and d['first_mismatch']is None
   assert len(d['rows'])==24 and all(v['passed']and v['output_finite']is True for v in d['rows'])
   assert sum(v for k,v in d['counts'].items()if k.endswith(':prefill'))==24 and not any(k.endswith(':decode')for k in d['counts'])
   assert d['update_binding_is_install_time_object']
  assert sum(x['original_calls']for x in a['selected_dispatch']['rows'])==24 and sum(x['original_calls']for x in b['selected_dispatch']['rows'])==0
  assert not r.get('error') and not r.get('cleanup_errors');return dict(passed=True,errors=[])
 except(AssertionError,KeyError,TypeError):return dict(passed=False,errors=['Indexed prefill functional HOLD'])
