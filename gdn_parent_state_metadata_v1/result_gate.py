def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and not r.get('error') and not r.get('cleanup_errors')
  assert r['adapter_restore'][0]['restored'] and r['precision_policy']['restored'] and r['precision_policy']['restored_split_k']
  assert r['public_logprob_gate']['exact_numeric'] and r['public_logprob_gate']['values']==2560
  a,b=r['shadow_request'],r['replacement_request'];assert a['output_token_ids']==b['output_token_ids'] and len(a['output_token_ids'])==32 and all(len(x)==16 for x in a['output_token_ids'])
  for arm in (a,b):
   assert arm['cross_runtime_tokens_equal'];assert arm['allow_bf16_reduced_precision_reduction']is True and arm['allow_bf16_reduced_precision_reduction_split_k']is True
   d=arm['selected_dispatch'];assert d['native_only'] and d['metadata_only'] and d['owner_payload_bytes']==0 and d['failed']is None and len(d['rows'])==24
   assert len({x['layer']for x in d['rows']})==24 and all(x['passed']and x['order']==['gather','invert','mask','core_enter','core_exit','scatter']for x in d['rows'])
  return dict(passed=True,scope='Two native requests,24 parent memory sequences each; no owner value preservation or TileIR implementation proof')
 except (AssertionError,KeyError,TypeError):return dict(passed=False,scope='Native metadata capture HOLD')
