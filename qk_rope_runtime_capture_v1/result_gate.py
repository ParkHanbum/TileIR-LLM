import hashlib
from pathlib import Path
def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['correctness_only'] and r['engine_options']['enforce_eager']
  rows=r['warmup_records']+r['samples'];assert len(r['warmup_records'])==len(r['samples'])==1
  assert all(x['generated_tokens']==512 and x['cross_runtime_tokens_equal'] is True and x['output_token_ids']==rows[0]['output_token_ids'] for x in rows)
  assert len(r['capture_manifest'])==1;m=r['capture_manifest'][0]
  assert m['complete'] and m['observers_removed'] and not m['errors'] and len(m['samples'])==2
  assert len(m['modules'])==8 and all(x['selected_fused'] for x in m['modules'].values())
  expected={n+'|'+phase:count for n in m['modules'] for phase,count in [('prefill',1),('decode',15)]};assert m['observed_calls']==expected
  assert hashlib.sha256(Path(m['manifest_path']).read_bytes()).hexdigest()==m['manifest_sha256']
  for s in m['samples']:
   assert s['selected_fused'] and s['original_call_count']==1 and s['return_preserved'] and s['project_return_preserved'] and s['gate_copy_raw_equal'] and s['outputs_fresh_disjoint'] and all(s['input_storage_preserved'].values())
   assert s['module_path'].endswith('layers.3.self_attn')
   for t in [*s['pre'].values(),*s['post'].values(),*s['outputs']]:
    for p,h in [('path','sha256'),('storage_path','storage_sha256')]:assert hashlib.sha256(Path(t[p]).read_bytes()).hexdigest()==t[h]
  assert {s['phase'] for s in m['samples']}=={'prefill','decode'} and not r.get('error')
  return dict(passed=True,errors=[])
 except (KeyError,AssertionError,TypeError,IndexError,OSError):return dict(passed=False,errors=['qk_helper_capture_contract'])
