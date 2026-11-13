def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['public_logprob_gate']['passed'] and r['adapter_restore'][0]['restored']
  assert r['precision_policy']['restored'] and r['precision_policy']['restored_split_k']
  assert r['precision_policy']['policy']=='unchanged native'
  assert r['precision_policy']['initial']==dict(bf16=True,split_k=True)
  a,b=r['shadow_request'],r['replacement_request']
  assert a['output_token_ids']==b['output_token_ids'] and len(a['output_token_ids'])==32 and all(len(t)==16 for t in a['output_token_ids'])
  for arm in (a,b):
   d=arm['selected_dispatch'];assert d['shadow_pass'] and d['supported_calls']==3184 and d['supported_vendor_fallback_calls']==0
   assert {k:v['supported_calls'] for k,v in d['components'].items()}=={'norm':1040,'attention_gate':128,'block_gather':16,'slot_mapping':16,'pos_seq':16,'gated_norm':384,'sample_counts':16,'qk_rope':128,'silu_mul':512,'indexed_decode_conv':360,'indexed_prefill_conv':24,'post_conv':24,'embedding':16,'attention_kv':128,'greedy_sample':16,'gdn_decode_core':360}
   for component in d['components'].values():
    assert len(component['rows'])==component['supported_calls']
    assert all(row.get('passed',True) for row in component['rows'])
   assert arm['allow_bf16_reduced_precision_reduction'] is True and arm['allow_bf16_reduced_precision_reduction_split_k'] is True
  for key,n in [('attention_gate',128),('block_gather',16),('slot_mapping',16),('sample_counts',16),('qk_rope',128),('silu_mul',512),('embedding',16),('attention_kv',128),('greedy_sample',16)]:
   assert a['selected_dispatch']['components'][key]['native_gate_calls']==n
   assert b['selected_dispatch']['components'][key]['native_gate_calls']==0
   assert b['selected_dispatch']['components'][key]['tileir_gate_calls']==n
  assert a['selected_dispatch']['components']['pos_seq']['native_calls']==16 and b['selected_dispatch']['components']['pos_seq']['native_calls']==0
  assert b['selected_dispatch']['components']['pos_seq']['tileir_calls']==16
  assert r['public_logprob_gate']['values']==2560 and r['public_logprob_gate']['exact_numeric'] and r['public_logprob_gate']['max_abs']==0
  ca=a['selected_dispatch']['components']['indexed_decode_conv'];cb=b['selected_dispatch']['components']['indexed_decode_conv']
  assert sum(v['original_calls'] for v in ca['rows'])==360 and sum(v['original_calls'] for v in cb['rows'])==0
  assert sum(v['tileir_calls'] for v in cb['rows'])==360
  for cc in (ca,cb):
   assert sum(v for k,v in cc['counts'].items() if k.endswith(':prefill'))==24
   assert sum(v for k,v in cc['counts'].items() if k.endswith(':decode'))==360
  pa=a['selected_dispatch']['components']['indexed_prefill_conv'];pb=b['selected_dispatch']['components']['indexed_prefill_conv']
  for pp in (pa,pb):
   assert pp['first_mismatch'] is None and pp['update_binding_is_install_time_object'] is False
   assert sum(v for k,v in pp['counts'].items() if k.endswith(':prefill'))==24 and not any(k.endswith(':decode') for k in pp['counts'])
   assert all(row['output_finite'] is True for row in pp['rows'])
  assert sum(row['original_calls'] for row in pa['rows'])==24 and sum(row['original_calls'] for row in pb['rows'])==0
  for arm,native in ((a,24),(b,0)):
   pc=arm['selected_dispatch']['components']['post_conv'];assert pc['native_gate_calls']==native and pc['tileir_gate_calls']==24
  assert r['adapter_restore'][0]['components']['indexed_prefill_conv']['update_binding_is_install_time_object'] is True
  for arm,original in ((a,360),(b,0)):
   core=arm['selected_dispatch']['components']['gdn_decode_core']
   assert core['other_bindings_unchanged'] and core['first_mismatch'] is None and len(core['counts'])==24 and set(core['counts'].values())=={15}
   assert sum(x['original_calls'] for x in core['rows'])==original and sum(x['tileir_calls'] for x in core['rows'])==360
  assert not r.get('cleanup_errors') and not r.get('error') and not r.get('cycles') and not r.get('warmup_records')
  return dict(passed=True,errors=[],scope='Sixteen-adapter functional composition only; no performance or ABI claim')
 except (KeyError,AssertionError,TypeError):return dict(passed=False,errors=['Composed functional contract HOLD'])
