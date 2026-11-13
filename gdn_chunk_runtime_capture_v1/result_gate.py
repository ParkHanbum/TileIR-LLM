def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['math_flags_unchanged']
  req=r['warmup_records']+r['samples'];assert len(req)==2 and all(x['generated_tokens']==512 and x['cross_runtime_tokens_equal'] is True for x in req)
  assert req[0]['output_token_ids']==req[1]['output_token_ids']
  m,=r['capture_manifest'];assert m['complete'] and m['observers_removed'] and len(m['counts'])==24 and all(x==1 for x in m['counts'].values())
  s,=m['samples'];assert s['original_calls']==1 and s['return_identity_preserved'] and set(s['outputs'])=={'o','final_state'}
  return dict(passed=True,errors=[])
 except (AssertionError,KeyError,ValueError,TypeError):return dict(passed=False,errors=['capture incomplete'])
