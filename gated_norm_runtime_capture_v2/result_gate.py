import hashlib
from pathlib import Path
def audit(r):
 errors=[]
 def need(x,k):
  if not x:errors.append(k)
 try:
  need(r.get('complete') is True and r.get('engine_shutdown') is True and r.get('correctness_only') is True,'completion')
  need(r['engine_options']['enforce_eager'] is True,'eager')
  rows=r['warmup_records']+r['samples'];need(len(r['warmup_records'])==1 and len(r['samples'])==1,'requests')
  for x in rows:need(x['generated_tokens']==512 and x['cross_runtime_tokens_equal'] is True and x['output_token_ids']==rows[0]['output_token_ids'],'tokens')
  ms=r['capture_manifest'];need(len(ms)==1,'worker');m=ms[0]
  need(m['complete'] is True and m['observers_removed'] is True,'capture')
  need(len(m['samples'])==2 and {(x['kind'],x['phase']) for x in m['samples']}=={('gated',p) for p in ('prefill','decode')},'four_cases')
  need(len(m['weights'])==1,'weights');ts=list(m['weights'].values())
  for w in ts:need(w['logical_sha256']==w['final_logical_sha256'],'weight')
  for x in m['samples']:
   need(len(x['outputs'])==1,'outputs')
   need(x['weight_post_logical_sha256']==m['weights'][x['weight_key']]['logical_sha256'],'weight_post')
   need(x['selected_forward']==m['modules'][x['kind']]['selected_forward'],'selected_forward')
   ts.extend(list(x['pre'].values())+list(x['post_inputs'].values())+x['outputs'])
  for t in ts:
   need(hashlib.sha256(Path(t['path']).read_bytes()).hexdigest()==t['sha256'],'file')
   meta=t['metadata'];need(type(meta['requires_grad']) is bool and type(meta['is_leaf']) is bool and isinstance(meta['python_type'],str) and bool(meta['python_type']),'original_tensor_descriptor')
  need(not r.get('error'),'error')
 except (KeyError,TypeError,IndexError,OSError,AttributeError):errors.append('malformed')
 return dict(passed=not errors,errors=errors)
