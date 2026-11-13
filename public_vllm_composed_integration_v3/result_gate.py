def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['public_logprob_gate']['passed'] and r['adapter_restore'][0]['restored']
  assert r['precision_policy']['restored'] and r['precision_policy']['restored_split_k']
  a,b=r['shadow_request'],r['replacement_request']
  assert a['output_token_ids']==b['output_token_ids'] and len(a['output_token_ids'])==32 and all(len(t)==16 for t in a['output_token_ids'])
  for arm in (a,b):
   d=arm['selected_dispatch'];assert d['shadow_pass'] and d['supported_calls']==1616 and d['supported_vendor_fallback_calls']==0
   assert {k:v['supported_calls'] for k,v in d['components'].items()}=={'norm':1040,'attention_gate':128,'block_gather':16,'slot_mapping':16,'pos_seq':16,'gated_norm':384,'sample_counts':16}
   for component in d['components'].values():
    assert len(component['rows'])==component['supported_calls']
    assert all(row.get('passed',True) for row in component['rows'])
   assert arm['allow_bf16_reduced_precision_reduction'] is False and arm['allow_bf16_reduced_precision_reduction_split_k'] is True
  for key,n in [('attention_gate',128),('block_gather',16),('slot_mapping',16),('sample_counts',16)]:
   assert a['selected_dispatch']['components'][key]['native_gate_calls']==n
   assert b['selected_dispatch']['components'][key]['native_gate_calls']==0
   assert b['selected_dispatch']['components'][key]['tileir_gate_calls']==n
  assert a['selected_dispatch']['components']['pos_seq']['native_calls']==16 and b['selected_dispatch']['components']['pos_seq']['native_calls']==0
  assert b['selected_dispatch']['components']['pos_seq']['tileir_calls']==16
  assert not r.get('error') and not r.get('cycles') and not r.get('warmup_records')
  return dict(passed=True,errors=[],scope='Seven-adapter functional composition only; no performance or ABI claim')
 except (KeyError,AssertionError,TypeError):return dict(passed=False,errors=['Composed functional contract HOLD'])
