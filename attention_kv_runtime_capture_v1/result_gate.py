def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['math_flags_unchanged']
  a,b=r['warmup_records']+r['samples'];assert a['output_token_ids']==b['output_token_ids'] and all(x['generated_tokens']==512 and x['cross_runtime_tokens_equal'] is True for x in (a,b))
  m,=r['capture_manifest'];assert m['complete'] and m['observers_removed'] and len(m['modules'])==8 and sum(x['count'] for x in m['counts'])==128
  assert {s['phase'] for s in m['samples']}=={'prefill','decode'}
  assert all(s['original_calls']==1 and s['return_is_none'] and all(s['inputs_preserved'].values()) for s in m['samples'])
  return dict(passed=True,errors=[])
 except (AssertionError,KeyError,TypeError,ValueError):return dict(passed=False,errors=['KV capture incomplete'])
