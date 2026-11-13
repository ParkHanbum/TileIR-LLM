from pathlib import Path
from residual_four_gdn_zero_lean_v3.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch as base_dispatch,audit as base_audit

def qualification(root):
 from .prior import verify
 prior=verify()
 p=Path(root).parent
 from residual_four_gdn_zero_lean_v3.gates import qualification as base
 from integer_pair_runtime_v1.gates import qualification as pair
 from runner_scalar32_lean_v1.gates import qualification as scalar
 from staged_typed_lean_runtime_v1.gates import qualification as staged
 from runtime_metadata_five_lean_v1.gates import qualification as metadata
 return dict(prior_extensions=prior,base=base(p/'residual_four_gdn_zero_lean_v3'),pair=pair(p/'integer_pair_runtime_v1'),scalar=scalar(p/'runner_scalar32_lean_v1'),staged=staged(p/'staged_typed_lean_runtime_v1'),metadata=metadata(p/'runtime_metadata_five_lean_v1'),whole_endpoint_pending=True)
def dispatch(d):
 r=base_dispatch(d);n=d['new_domains']
 f=d['finite504'];assert f['calls']==504 and f['dense']==376 and f['strided']==128 and f['native_predicates']==0 and f['tile_predicates']==504
 assert f['counts']=={**{'post.'+k:24 for k in ('conv_output','a','b','A_log','dt_bias')},'kv.key':128,'kv.value':128,'attention.out':128}
 q=d['eq408'];assert q['calls']==408 and q['counts']==q['returns']=={'attention.q':128,'attention.k':128,'attention.v':128,'gdn.initial':24} and q['native_predicates']==0 and q['failed'] is None
 c=n['batch_cast'];assert c==dict(calls=1,native_production_copy=0,native_private_oracle=0,native_same_dtype_H2D=1,tileir_cast_launches=1,temporary_bytes=1792,raw_rechecked=False,owners_released=True)
 assert n['metadata_batch_joint']['metadata']==n['metadata'] and n['metadata_batch_joint']['batch_cast']==c and n['metadata_batch_joint']['logical_calls']==6
 from integer_pair_runtime_v1.gates import pair_gate
 assert pair_gate(n['integer_pair'],'replace')
 assert n['scalar']==dict(calls=32,attempts=32,indices=list(range(32)),failed=None,selected_native_calls=0,selected_fallback_calls=0,logical_operations=32)
 m=n['metadata'];assert m['calls']==m['candidate_calls']==5 and m['native_oracle_calls']==0 and m['counts']==dict(subq=1,subs=1,gt=1,full=2)
 return r

def audit(r,root=None):
 root=Path(root) if root is not None else Path(__file__).resolve().parent
 b=base_audit(r,root)
 if not b['passed']:return b
 try:
  from integer_pair_runtime_v1.gates import pair_gate
  observed=r['native_request']['pair_observation'];assert observed['current_run_native_observed']is True and observed['current_run_raw_comparison']is False
  assert pair_gate(observed['integer_pair'],'shadow')
  dispatch(r['lean_request']['selected_dispatch']);assert r['extensions_restored']and r['extension_owners_released']and not r.get('extension_cleanup_errors')
  o=r['staged_ownership'];assert o['first_failure']is None and o['released'];rows=o['replacement_rows'];assert len(rows)==140
  from staged_typed_shadow_runtime_v2.evidence import check_rows
  check_rows(rows,o['counts']);assert sum(e['launch_returns']for x in rows for e in x['launch_ledger'])==9
  for label,n in [('staged',123),('fused',17)]:
   c=o['counts'];key='request.'+label;assert c[key+'.calls']==c[key+'.candidate_attempts']==c[key+'.candidate_returns']==n and c.get(key+'.native_attempts',0)==c.get(key+'.native_returns',0)==0
  assert r['new_domain_bindings']['metadata_batch_joint_active'] and r['new_domain_bindings']['scalar_owner_verified'] and r['new_domain_bindings']['integer_installed_before_base']
  return dict(passed=True,new_domains=dict(integer=64,scalar=32,staged=140,metadata=5),whole_endpoint=True,physical_coverage_unmeasured=True,performance=False)
 except (AssertionError,KeyError,TypeError,ValueError):return dict(passed=False,scope='whole endpoint extension')
