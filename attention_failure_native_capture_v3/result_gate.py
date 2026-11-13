def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['math_flags_unchanged'] and not r.get('error') and not r.get('cleanup_errors')
  req=r['warmup_records']+r['samples'];assert len(req)==2 and all(x['generated_tokens']==512 and x['cross_runtime_tokens_equal'] is True and x['output_token_ids']==req[0]['output_token_ids'] for x in req)
  m,=r['capture_manifest'];assert m['complete'] and m['observers_removed'] and len(m['calls'])==128 and len(m['samples'])==3
  assert {s['phase'] for s in m['samples']}=={'prefill','decode'} and all(s['inputs_preserved'] and s['helper_return_identity_preserved'] and s['finite_output'] for s in m['samples'])
  assert r['initial_math_flags']==[True,True] and not m['failed']
  assert {s['capture_label'] for s in m['samples']}=={'second_attention_call','late_65','late_68'}
  failed,=[s for s in m['samples'] if s['capture_label']=='second_attention_call']
  assert failed['call_ordinal']==1 and failed['prior_q_logical_match'] and failed['prior_native_logical_match']
  return dict(passed=True,errors=[],scope='Native capture provenance only; prior candidate HOLD remains')
 except (AssertionError,KeyError,TypeError,ValueError):return dict(passed=False,errors=['Attention core native capture HOLD'])
