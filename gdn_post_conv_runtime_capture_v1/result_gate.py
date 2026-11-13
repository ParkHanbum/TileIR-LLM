def audit(r):
 errors=[]
 try:
  assert r['complete'] and r['engine_shutdown'] and r['math_flags_unchanged']
  req=r['warmup_records']+r['samples'];assert len(req)==2
  assert all(x['generated_tokens']==512 and x['cross_runtime_tokens_equal'] is True and x['output_token_ids']==req[0]['output_token_ids'] for x in req)
  m,=r['capture_manifest'];assert m['complete'] and m['observers_removed'] and len(m['modules'])==24
  assert set(m['counts'])==set(m['modules']) and all(v==1 for v in m['counts'].values())
  assert all(v['preserved'] for v in m['parameter_checks'].values())
  s,=m['samples'];assert s['original_calls']==1 and s['return_identity_preserved']
  assert s['phase']['phase']=='prefill' and all(s['input_storage_preserved'].values())
  assert set(s['outputs'])=={'q','k','v','g','beta'}
 except (AssertionError,KeyError,ValueError,TypeError):errors.append('contract_or_completion')
 return dict(passed=not errors,errors=errors)
