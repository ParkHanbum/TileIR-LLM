import json
from pathlib import Path
from lean_public_runtime_qualification_v2.gates import sha,frozen_reference,exact_tokens,exact_public
# Kept import-compatible with shared runner helpers, never used for this native diagnostic.
def dispatch(*args):raise AssertionError('No TileIR replacement dispatch in native diagnostic')
def qualification(root):
 root=Path(root);pins=json.loads((root/'SOURCE-PINS.json').read_text())
 for p,h in pins.items():assert sha(p)==h
 refs=json.loads((root/'PREREQUISITE.json').read_text());data={}
 for kind,item in refs.items():
  assert sha(item['path'])==item['sha256'];data[kind]=json.loads(Path(item['path']).read_text())
 from staged_ownership_runtime_v1.gates import audit as prior_audit
 r,l,c=data['result'],data['launch'],data['recovery']
 assert prior_audit(r,Path(refs['result']['path']).parent.parent)['passed']
 assert l['returncode']==c['actual_returncode']==0 and l['qualification_complete']and c['child_absent']and c['lock_available']and not c['remaining_process_group']
 return dict(source_verified=True,prior_actual_ownership=refs,scope='Actual typed-owner observation; clone numerical proof pending')
def audit(r,root=None):
 try:
  root=Path(root or Path(__file__).parent)
  assert r['complete']and r['candidate_checks_passed']and not r.get('error')and not r.get('cleanup_errors')
  assert r['timed_requests']==0 and r['scope']=='native-only staged ownership diagnostic'
  ft,fp,opts=frozen_reference();assert r['engine_options']==opts
  for name in ('native_request','observed_request'):
   x=r[name];exact_tokens(x['output_token_ids'],ft);exact_public(x['public_logprobs'],fp)
   assert x['selected_dispatch']is None and x['allow_bf16_reduced_precision_reduction']is True and x['allow_bf16_reduced_precision_reduction_split_k']is True
  assert r['precision_policy']['initial']==dict(bf16=True,split_k=True)and r['precision_policy']['restored']and r['precision_policy']['restored_split_k']
  assert len(r['method_bindings'])==4 and all(x['source_verified']for x in r['method_bindings'])
  o=r['ownership'];c=o['counts'];assert o['first_failure']is None and o['released']and o['actual_write_gpu_count']is None
  for label in ('staged','fused'):
   key='request.'+label;n=c.get(key+'.calls',0);assert n>0 and c[key+'.native_attempts']==c[key+'.native_returns']==c[key+'.stream_unchanged']==c[key+'.alias_preserved']==n
  for key,n in c.items():
   if key.endswith('.calls'):
    base=key[:-6];assert c.get(base+'.native_attempts')==c.get(base+'.native_returns')==n
  assert r['observer_restored']and r['synchronized_after_restore']and r['engine_shutdown']
  assert r['cache_preseed']['destination']==str(root/'cache_v1')and r['cache_preseed']['files_verified']and r['cache_preseed']['separate_inodes_verified']
  rows=o['shadow_rows'];assert len(rows)==140 and all(x['passed']and x['native_returns']==1 and x['candidate_returns']==1 and x['raw_equal']and x['native_list_semantics']for x in rows)
  from .evidence import check_rows
  check_rows(rows,c)
  return dict(passed=True,scope='Same-input clone TileIR raw comparison; original production writes/returns native, no live replacement/performance proof')
 except (AssertionError,KeyError,TypeError,ValueError)as e:return dict(passed=False,error=type(e).__name__)
