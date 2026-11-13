import hashlib
from pathlib import Path
def audit(r):
 errors=[]
 try:
  assert r['complete'] is True and r['engine_shutdown'] is True and r['correctness_only'] is True and r['engine_options']['enforce_eager'] is True
  rows=r['warmup_records']+r['samples'];assert len(r['warmup_records'])==len(r['samples'])==1
  assert all(x['generated_tokens']==512 and x['cross_runtime_tokens_equal'] is True and x['output_token_ids']==rows[0]['output_token_ids'] for x in rows)
  assert len(r['capture_manifest'])==1;m=r['capture_manifest'][0];assert m['complete'] is True and m['observers_removed'] is True and len(m['samples'])==2
  assert {s['phase'] for s in m['samples']}=={'prefill','decode'}
  for s in m['samples']:
   assert s['association']==['project_return_gate','attention_return','o_proj_input'] and not any(s['input_mutated'].values())
   for t in [*s['pre'].values(),*s['post_inputs'].values(),*s['outputs']]:
    assert hashlib.sha256(Path(t['path']).read_bytes()).hexdigest()==t['sha256']
    assert type(t['metadata']['requires_grad']) is bool
  assert not r.get('error')
 except (KeyError,AssertionError,TypeError,IndexError,OSError):errors.append('capture_contract')
 return dict(passed=not errors,errors=errors)
