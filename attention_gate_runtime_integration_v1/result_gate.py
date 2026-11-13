def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['public_logprob_gate']['passed']
  assert r['precision_policy']['restored'] and r['precision_policy']['restored_split_k']
  assert r['adapter_restore'][0]['restored']
  a,b=r['shadow_request'],r['replacement_request']
  assert a['selected_dispatch']['shadow_pass'] and a['selected_dispatch']['supported_calls']==b['selected_dispatch']['supported_calls']==128
  assert a['selected_dispatch']['native_gate_calls']==128 and b['selected_dispatch']['native_gate_calls']==0
  assert a['selected_dispatch']['tileir_gate_calls']==b['selected_dispatch']['tileir_gate_calls']==128
  assert a['output_token_ids']==b['output_token_ids'] and len(a['output_token_ids'])==32 and all(len(t)==16 for t in a['output_token_ids'])
  assert a['allow_bf16_reduced_precision_reduction'] is b['allow_bf16_reduced_precision_reduction'] is False
  assert a['allow_bf16_reduced_precision_reduction_split_k'] is b['allow_bf16_reduced_precision_reduction_split_k'] is True
  assert not r.get('error') and not r.get('cycles') and not r.get('warmup_records')
  return dict(passed=True,errors=[],scope='Attention gate functional correctness only, no performance')
 except (AssertionError,KeyError,TypeError):return dict(passed=False,errors=['Incomplete attention gate shadow/replacement correctness'])
