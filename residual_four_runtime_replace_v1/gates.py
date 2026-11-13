import json
from pathlib import Path
from lean_public_runtime_qualification_v2.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch
from lean_public_runtime_qualification_v2.gates import qualification as lean_qualification,audit as endpoint_audit
from residual_rope_combine_bias_port_v4.gates import audit as shadow_audit
P=Path(__file__).resolve().parent.parent
def qualification(root):
 lean=lean_qualification(P/'lean_public_runtime_qualification_v2');q=P/'residual_rope_combine_bias_port_v4'
 paths=[q/'residualFourShadow02/RESULT.json',q/'residualFourShadow02-launch.json',q/'residualFourShadow02-recovery.json']
 hashes=['5c5dae4d58c55e02eb90cf404e67c7d05fd612e5c05486caec8ce0eef8865977','a2ab965de239b4c34c9e073837aba585b054d406485bb8b5db629c772bb16f41','a297f862244758c31ab81a2a2ebf2f23fc29e35cd1e3a1a2f5f7d9b94e32dd79']
 assert all(sha(p)==h for p,h in zip(paths,hashes));r,l,c=[json.loads(p.read_text())for p in paths]
 assert shadow_audit(r,q)['passed']and l['returncode']==0 and l['qualification_complete']and c['child_absent']and not c['remaining_process_group']and c['lock_available']
 return dict(lean=lean,clone_proof=dict(zip(map(str,paths),hashes)),limit='Worker child exit0 durable; external outer-shell exit not separately retained')
def audit(r,root=None):
 root=Path(root or Path(__file__).resolve().parent);a=endpoint_audit(r,root)
 if not a['passed']:return a
 try:
  assert r['four_binding']==[dict(shared_latch=True,owned_copy_identity=True,per_helper_cap=16)]
  d=r['four_replacement'];assert d['complete']and d['restored']and not d['failed']
  assert d['calls']==dict(prepare_rope_positions=16,combine_sampled_and_draft_tokens=16,bias=16,post_update=16)
  assert len(d['rows'])==64 and all(x['passed']and x['native_calls']==0 and x['candidate_calls']==1 and x['original_addresses_preserved']and x['preflight_passed']for x in d['rows'])
  assert r['uva_constructor_restored']and r['uva_release']['released']and r['uva_release']['after_synchronization']and not r.get('four_cleanup_errors')
  return dict(passed=True,scope='64 live-address replacements/CPU raw oracle + lean25 endpoints; correctness only')
 except (AssertionError,KeyError,TypeError)as e:return dict(passed=False,errors=[repr(e)])
