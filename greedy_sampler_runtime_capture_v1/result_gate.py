def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['math_flags_unchanged'] and not r.get('cleanup_errors')
  req=r['warmup_records']+r['samples'];assert len(req)==2
  assert all(x['generated_tokens']==512 and x['cross_runtime_tokens_equal'] is True and x['output_token_ids']==req[0]['output_token_ids'] for x in req)
  assert r['public_logprob_gate']['exact_numeric'] and r['public_logprob_gate']['values']==2560
  m,=r['capture_manifest'];assert m['complete'] and m['observers_removed'] and len(m['rows'])==16
  assert [x['index'] for x in m['rows']]==list(range(16))
  assert all(x['original_calls']==1 and x['counts']=={'gumbel':1,'topk':1} and x['passed'] for x in m['rows'])
  assert {(s['call_index'],s['kind']) for s in m['samples']}=={(i,k) for i in (0,15) for k in ('gumbel','topk')}
  assert all(all(s['observations']['input_logical_preserved'].values()) for s in m['samples'])
  return dict(passed=True,errors=[])
 except (KeyError,AssertionError,TypeError,ValueError):return dict(passed=False,errors=['Sampler capture HOLD'])
