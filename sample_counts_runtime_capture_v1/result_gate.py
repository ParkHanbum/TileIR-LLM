import hashlib
from pathlib import Path
def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and r['correctness_only'] and r['engine_options']['enforce_eager']
  rows=r['warmup_records']+r['samples'];assert len(r['warmup_records'])==len(r['samples'])==1
  assert all(x['generated_tokens']==512 and x['cross_runtime_tokens_equal'] is True and x['output_token_ids']==rows[0]['output_token_ids'] for x in rows)
  assert len(r['capture_manifest'])==1;m=r['capture_manifest'][0]
  assert m['complete'] and m['observers_removed'] and not m['errors'] and len(m['samples'])==2 and {s['phase'] for s in m['samples']}=={'prefill','decode'}
  assert hashlib.sha256(Path(m['manifest_path']).read_bytes()).hexdigest()==m['manifest_sha256']
  for s in m['samples']:
   assert s['original_call_count']==1 and s['return_preserved'] and s['first_return_same_object'] and s['rejected_disjoint'] and all(s['input_storage_preserved'].values())
   flags=s['batch']['is_prefilling'];assert flags and (all(flags) if s['phase']=='prefill' else not any(flags))
   for t in [*s['pre'].values(),*s['post'].values(),*s['outputs']]:
    for p,h in [('path','sha256'),('storage_path','storage_sha256')]:assert hashlib.sha256(Path(t[p]).read_bytes()).hexdigest()==t[h]
  assert not r.get('error')
  return dict(passed=True,errors=[])
 except (KeyError,AssertionError,TypeError,IndexError,OSError):return dict(passed=False,errors=['capture_contract'])
