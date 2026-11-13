import json
from pathlib import Path
from lean26_gdn_finite_runtime_v1.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch,audit as endpoint_audit
P=Path(__file__).resolve().parent.parent
def qualification(root):
 from residual_four_runtime_replace_v1.gates import audit as four_audit
 refs=json.loads((Path(root)/'PREREQUISITE.json').read_text());assert set(refs)=={'four','lean26'}
 for label,items in refs.items():
  data={}
  for kind,item in items.items():
   assert sha(item['path'])==item['sha256'];data[kind]=json.loads(Path(item['path']).read_text())
  r,l,c=data['result'],data['launch'],data['recovery']
  audit=four_audit if label=='four'else endpoint_audit
  assert audit(r,Path(items['result']['path']).parent.parent)['passed']
  assert l['returncode']==c['actual_returncode']==0 and l['qualification_complete']
  assert c['child_absent']and not c['remaining_process_group']and c['lock_available']
  assert l['result_sha256']==c['result_sha256']==items['result']['sha256'] and c['launch_sha256']==items['launch']['sha256']
 return dict(passed=True,evidence=refs,scope='Separate live-address four and lean26+472 guards; combined proof pending')
def audit(r,root=None):
 root=Path(root or Path(__file__).resolve().parent);a=endpoint_audit(r,root)
 if not a['passed']:return a
 try:
  assert r['four_binding']==[dict(shared_latch=True,owned_copy_identity=True,per_helper_cap=16)]
  d=r['four_replacement'];assert d['complete']and d['restored']and not d['failed']
  assert d['calls']==dict(prepare_rope_positions=16,combine_sampled_and_draft_tokens=16,bias=16,post_update=16)
  assert len(d['rows'])==64 and all(x['passed']and x['native_calls']==0 and x['candidate_calls']==1 and x['original_addresses_preserved']and x['preflight_passed']and x['output_raw']and x['readonly_raw']and x['return_contract']for x in d['rows'])
  assert r['uva_constructor_restored']and r['uva_release']['released']and r['uva_release']['after_synchronization']and not r.get('four_cleanup_errors')
  return dict(passed=True,scope='64 live-address replacements/CPU raw oracle + lean26 endpoints and472 finite guards; correctness only')
 except (AssertionError,KeyError,TypeError)as e:return dict(passed=False,errors=[repr(e)])
