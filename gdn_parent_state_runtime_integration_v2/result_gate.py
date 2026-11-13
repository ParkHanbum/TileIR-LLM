def native_finish_gate(d):
 try:
  assert d['native_only'] is False and d['metadata_only'] is False and d['core_native'] is True and d['failed'] is None
  rows=d['rows'];assert len(rows)==24 and len({x['layer']for x in rows})==24
  assert all(x['passed'] is True and x['order']==['gather','invert','mask','core_enter','core_exit','scatter']for x in rows)
  assert all(x.get('gather_raw') and x.get('mask_raw') and x.get('scatter_raw') and x.get('final_input_hash_preserved') and x.get('unselected_padding_hash_preserved') and x.get('selected_restore_hash_preserved') and x.get('native_oracle_calls')==3 and x.get('tile_calls')==3 for x in rows)
  return True
 except (AssertionError,KeyError,TypeError):return False
def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and not r.get('error') and not r.get('cleanup_errors')
  assert r['adapter_restore'][0]['restored'] and r['precision_policy']['restored'] and r['precision_policy']['restored_split_k']
  assert r['public_logprob_gate']['exact_numeric'] and r['public_logprob_gate']['values']==2560
  a,b=r['shadow_request'],r['replacement_request'];assert a['output_token_ids']==b['output_token_ids'] and len(a['output_token_ids'])==32 and all(len(x)==16 for x in a['output_token_ids'])
  for arm in (a,b):
   assert arm['cross_runtime_tokens_equal'];assert arm['allow_bf16_reduced_precision_reduction']is True and arm['allow_bf16_reduced_precision_reduction_split_k']is True
   assert native_finish_gate(arm['selected_dispatch'])
  return dict(passed=True,scope='24 shadow then24 conditional-return replacement parent sequences; core native, selected raw, untouched view hash only; diagnostic oracles retained')
 except (AssertionError,KeyError,TypeError):return dict(passed=False,scope='Native metadata capture HOLD')
