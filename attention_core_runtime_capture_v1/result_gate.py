def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['math_flags_unchanged'] and not r.get('error') and not r.get('cleanup_errors')
  req=r['warmup_records']+r['samples'];assert len(req)==2 and all(x['generated_tokens']==512 and x['cross_runtime_tokens_equal'] is True and x['output_token_ids']==req[0]['output_token_ids'] for x in req)
  m,=r['capture_manifest'];assert m['complete'] and m['observers_removed'] and len(m['calls'])==128 and len(m['samples'])==2
  assert {s['phase'] for s in m['samples']}=={'prefill','decode'} and all(s['inputs_preserved'] and s['helper_return_identity_preserved'] and s['finite_output'] for s in m['samples'])
  return dict(passed=True,errors=[])
 except (AssertionError,KeyError,TypeError,ValueError):return dict(passed=False,errors=['Attention core native capture HOLD'])
