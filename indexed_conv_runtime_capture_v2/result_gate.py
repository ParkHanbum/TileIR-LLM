def audit(r):
 errors=[]
 try:
  assert r['complete'] and r['engine_shutdown'] and r['math_flags_unchanged']
  req=r['warmup_records']+r['samples'];assert len(req)==2
  assert all(t['generated_tokens']==512 and t['output_token_ids']==req[0]['output_token_ids'] and t['cross_runtime_tokens_equal'] is True for t in req)
  m,=r['capture_manifest'];assert m['complete'] and m['observers_removed'] and len(m['modules'])==24
  assert {s['phase'] for s in m['samples']}=={'prefill','decode'}
  assert all(s['original_calls']==1 and s['state_unselected_unchanged'] and s['weight_unchanged'] for s in m['samples'])
 except (AssertionError,KeyError,ValueError,TypeError):errors.append('contract_or_completion')
 return dict(passed=not errors,errors=errors)
