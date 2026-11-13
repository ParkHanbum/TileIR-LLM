from pathlib import Path
from residual_four_gdn_zero_lean_v3.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch as base_dispatch,audit as base_audit

def qualification(root):
 p=Path(root).parent
 from residual_four_gdn_zero_lean_v3.gates import qualification as base
 from integer_pair_runtime_v1.gates import qualification as pair
 from runner_scalar32_lean_v1.gates import qualification as scalar
 from staged_typed_lean_runtime_v1.gates import qualification as staged
 from runtime_metadata_five_lean_v1.gates import qualification as metadata
 return dict(base=base(p/'residual_four_gdn_zero_lean_v3'),pair=pair(p/'integer_pair_runtime_v1'),scalar=scalar(p/'runner_scalar32_lean_v1'),staged=staged(p/'staged_typed_lean_runtime_v1'),metadata=metadata(p/'runtime_metadata_five_lean_v1'),whole_endpoint_pending=True)
def dispatch(d):
 r=base_dispatch(d);n=d['new_domains']
 from integer_pair_runtime_v1.gates import pair_gate
 assert pair_gate(n['integer_pair'],'replace')
 assert n['scalar']==dict(calls=32,attempts=32,indices=list(range(32)),failed=None,selected_native_calls=0,selected_fallback_calls=0,logical_operations=32)
 m=n['metadata'];assert m['calls']==m['candidate_calls']==5 and m['native_oracle_calls']==0 and m['counts']==dict(subq=1,subs=1,gt=1,full=2)
 return r

def audit(r,root=None):
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
  assert r['new_domain_bindings']['scalar_owner_verified'] and r['new_domain_bindings']['integer_installed_before_base']
  return dict(passed=True,new_domains=dict(integer=64,scalar=32,staged=140,metadata=5),whole_endpoint=True,physical_coverage_unmeasured=True,performance=False)
 except (AssertionError,KeyError,TypeError,ValueError):return dict(passed=False,scope='whole endpoint extension')
