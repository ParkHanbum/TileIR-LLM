def audit(r):
 try:
  assert r['complete']and r['engine_shutdown']and r['math_flags_unchanged']and not r.get('cleanup_errors')
  req=r['warmup_records']+r['samples'];assert len(req)==2
  assert all(x['generated_tokens']==512 and x['cross_runtime_tokens_equal']is True and x['output_token_ids']==req[0]['output_token_ids']for x in req)
  m,=r['capture_manifest'];assert m['complete']and m['observers_removed']and m['counts']=={'prefill':1,'decode':15}
  assert len(m['rows'])==16 and all(x['original_calls']==1 and x['passed']for x in m['rows'])
  assert len(m['samples'])==2 and all(x['raw_rows_equal']and x['selected_rows_preserved']and x['finite']for x in m['samples'])
  return {'passed':True,'errors':[]}
 except (KeyError,AssertionError,TypeError,ValueError):return {'passed':False,'errors':['capture HOLD']}
