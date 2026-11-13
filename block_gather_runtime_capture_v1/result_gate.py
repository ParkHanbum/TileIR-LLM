import hashlib
from pathlib import Path
def audit(r):
 errors=[]
 try:
  assert r['complete'] is True and r['engine_shutdown'] is True and r['correctness_only'] is True and r['engine_options']['enforce_eager'] is True
  assert len(r['warmup_records'])==len(r['samples'])==1
  rows=r['warmup_records']+r['samples']
  assert all(x['generated_tokens']==512 and x['cross_runtime_tokens_equal'] is True and x['output_token_ids']==rows[0]['output_token_ids'] for x in rows)
  assert len(r['capture_manifest'])==1;m=r['capture_manifest'][0]
  assert m['complete'] is True and m['observers_removed'] is True and not m['errors']
  assert len(m['samples'])==2 and {s['phase'] for s in m['samples']}=={'prefill','decode'}
  assert hashlib.sha256(Path(m['manifest_path']).read_bytes()).hexdigest()==m['manifest_sha256']
  for s in m['samples']:
   assert s['original_call_count']==1 and s['return_preserved'] is True and all(s['input_storage_preserved'].values())
   b=s['batch'];flags=b['is_prefilling']
   assert flags and (all(flags) if s['phase']=='prefill' else not any(flags))
   assert s['counts_placement']['device_view_shares_active_storage'] is True
   for t in [*s['pre'].values(),*s['post'].values(),*s['outputs']]:
    for p,h in [('path','sha256'),('storage_path','storage_sha256')]:assert hashlib.sha256(Path(t[p]).read_bytes()).hexdigest()==t[h]
  assert not r.get('error')
 except (KeyError,AssertionError,TypeError,IndexError,OSError):errors.append('capture_contract')
 return dict(passed=not errors,errors=errors)
