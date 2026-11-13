def native_finish_gate(d):
 try:
  assert d['shadow_only'] is False and d['replacement'] is (d['mode']=='replace') and d['failed'] is None
  assert len(d['rows'])==1
  r=d['rows'][0];assert r['passed'] and r['native_calls']==(1 if d['mode']=='shadow' else 0) and r['tile_calls']==(0 if d['mode']=='shadow' else 1) and r['outputs_cpu_reference_raw'] and r['readonly_raw'] and r['original_output_addresses'] and r['native_oracle_calls']==0
  return True
 except (KeyError,AssertionError,TypeError):return False
def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and not r.get('error') and not r.get('cleanup_errors')
  assert r['adapter_restore'][0]['restored'] and r['precision_policy']['restored'] and r['precision_policy']['restored_split_k']
  assert r['public_logprob_gate']['exact_numeric'] and r['public_logprob_gate']['values']==2560
  a,b=r['shadow_request'],r['replacement_request'];assert a['output_token_ids']==b['output_token_ids'] and len(a['output_token_ids'])==32 and all(len(x)==16 for x in a['output_token_ids'])
  for arm in (a,b):
   assert arm['cross_runtime_tokens_equal'];assert arm['allow_bf16_reduced_precision_reduction']is True and arm['allow_bf16_reduced_precision_reduction_split_k']is True
   assert native_finish_gate(arm['selected_dispatch'])
  return dict(passed=True,scope='Native request then TileIR actual original-output replacement; public CPU-source raw oracle and endpoints; no timing')
 except (AssertionError,KeyError,TypeError):return dict(passed=False,scope='Native metadata capture HOLD')
