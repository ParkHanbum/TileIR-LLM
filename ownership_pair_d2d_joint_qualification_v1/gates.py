import json
from pathlib import Path
from whole_lean_extensions_runtime_v5.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch,audit as base_audit

def qualification(root):
 root=Path(root);q=json.loads((root/'PREREQUISITE.json').read_text());assert q['complete'] is True,'CPU preparation pending'
 e={}
 for k,v in q['evidence'].items():assert sha(v['path'])==v['sha256'];e[k]=json.loads(Path(v['path']).read_text())
 old=root.parent/'whole_lean_extensions_runtime_v5';assert base_audit(e['result'],old)['passed']
 l,z=e['launch'],e['recovery'];assert l['returncode']==z['actual_returncode']==0 and l['child_absent'] and z['child_absent'] and not l['remaining_process_group'] and not z['remaining_process_group'] and l['lock_reacquired'] and z['lock_available']
 assert l['result_sha256']==z['result_sha256']==q['evidence']['result']['sha256']
 for p,h in json.loads((root/'PINS.json').read_text()).items():assert sha(p)==h,p
 return dict(prior_v5_actual=True,joint_raw_pending=True)
def audit(r,root=None):
 root=Path(root or Path(__file__).parent)
 try:
  assert base_audit(r,root)['passed']
  assert r['pair_synthetic']==dict(cases=48,equal_cases=8,unequal_cases=40,all_inputs_readonly=True,passed=True)
  p=r['pair_shadow'];assert p['table_checks']==34 and p['pair_checks']==68 and p['native_diagnostic_oracles']==68 and p['pointer_values_serialized'] is False
  assert len(p['rows'])==len(p['actual_metadata'])==68
  assert sum(x['dtype']=='torch.uint64'for x in p['rows'])==34 and sum(x['dtype']=='torch.int64'for x in p['rows'])==34
  assert all(x['native'] is True and x['candidate'] is True and x['whole_inputs_readonly'] is True for x in p['rows'])
  for row,obs in zip(p['rows'],p['actual_metadata']):
   for t in (obs['a'],obs['b']):
    assert t['dtype']==row['dtype'] and t['shape']==row['shape'] and t['stride']==[1] and t['offset']==0 and t['storage_bytes']==8*row['shape'][0]
    assert t['device'].startswith('cuda') and t['layout']=='torch.strided' and t['requires_grad'] is False and t['is_neg'] is False and t['is_conj'] is False
  assert p['actual_callback_origin']==r['pair_callback_origin'] and p['actual_callback_origin']['code_identity_verified'] is True and p['actual_callback_origin']['torch_closure_identity_verified'] is True
  c=r['copy_shadow'];assert c['complete'] is True and c['calls']==24 and len(c['rows'])==24 and c['model_copies_native'] is True and c['actual_replacement'] is False
  assert len({x['layer']for x in c['rows']})==24
  for x in c['rows']:
   assert x['native_calls']==x['candidate_calls']==1
   assert all(x[k] is True for k in ('raw_exact','outside_preserved','source_readonly','poison_independent'))
  assert r['copy_shadow_restored'] is True and r['pair_shadow_restored']==dict(restored=True,original_callback_identity=True)
  return dict(passed=True,scope='v5 endpoints plus same-input pair68/copy24 raw; diagnostic natives retained')
 except (AssertionError,KeyError,TypeError,ValueError)as e:return dict(passed=False,error=repr(e))
