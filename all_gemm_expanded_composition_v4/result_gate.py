"""Reuse unchanged standalone numerical gates on composition projections."""
from pathlib import Path
from .adapter import ORDER,SIXTEEN
from .gemm_gate import audit as gemm_audit
from .sixteen_gate import audit as sixteen_audit
from .gates import AUDITS
from .parent_contract import parent_gate
R=Path(__file__).resolve().parent

def project(r,key):
 out=dict(r);out['adapter_restore']=[dict(r['adapter_restore'][0])]
 for arm in ('shadow_request','replacement_request'):
  a=dict(r[arm]);parts=a['selected_dispatch']['components']
  if key=='sixteen':
   selected={k:parts[k] for k,m,n in SIXTEEN}
   a['selected_dispatch']=dict(components=selected,shadow_pass=all(v['shadow_pass'] for v in selected.values()),supported_calls=sum(v['supported_calls'] for v in selected.values()),supported_vendor_fallback_calls=sum(v.get('supported_vendor_fallback_calls',0) for v in selected.values()))
  else:a['selected_dispatch']=parts[key]
  out[arm]=a
 if key=='all_gemm':out['adapter_install']=[r['adapter_install'][0]['components'][key]]
 return out

def audit(r):
 try:
  assert r['complete'] and not r.get('error') and not r.get('cleanup_errors')
  c=r['cache_preseed'];assert c['destination']==str(R/'cache_v1') and c['files_verified'] and c['separate_inodes_verified']
  install=r['adapter_install'][0];assert install['install_order']==list(ORDER)
  assert set(install['entry_guard_map'])==set(ORDER) and all(any(x['kind']=='installed_binding' for x in rows) for rows in install['entry_guard_map'].values())
  assert install['controller_count']==25 and install['selected_boundary_calls_expected']==5888 and install['selected_logical_operations_expected']==6056
  restore=r['adapter_restore'][0];assert restore['restored'] and restore['save_guards_restored'] and restore['entry_guards_restored'] and restore['close_order']==list(reversed(ORDER))
  assert set(restore['components'])==set(ORDER) and all(v['restored'] for v in restore['components'].values())
  for phase in ('shadow_request','replacement_request'):
   d=r[phase]['selected_dispatch'];assert set(d['components'])==set(ORDER) and not d['failed'] and d['shadow_pass']
   assert d['selected_boundary_calls']==5888 and d['selected_logical_operations']==6056 and d['selected_vendor_fallback_calls']==0 and d['installation_order']==list(ORDER)
  details=dict(sixteen=sixteen_audit(project(r,'sixteen')),gemm=gemm_audit(project(r,'all_gemm'),R))
  for key in ('sampler_four','gdn_prefill_core','attention_core','runner_padding','hidden_gather','sampler_fp32','accepted_scatter'):details[key]=AUDITS[key](project(r,key))
  parent_details=[]
  for arm in ('shadow_request','replacement_request'):
   parts=r[arm]['selected_dispatch']['components'];d=parts['gdn_parent_state'];core=parts['gdn_prefill_core']
   assert d['shadow_pass'] and parent_gate(d)
   assert d['core_native_calls']==core['native_gate_calls'] and d['core_tileir_calls']==core['tileir_gate_calls']
   assert {x['path'] for x in d['core_phase_rows']}=={x['path'] for x in core['rows']}
   assert all(x['passed'] and x['phase']=='prefill' for x in core['rows'])
   parent_details.append(dict(passed=True,phase=d['mode'],parent_api_calls=96))
  assert restore['components']['gdn_parent_state']['core_guard_restored']
  details['parent_state']=dict(passed=True,phases=parent_details)
  assert all(v['passed'] for v in details.values()),details
  return dict(passed=True,errors=[],components=details,scope='6056 logical operations / 5888 boundaries / 25 controllers; not kernel launches or100% workload support')
 except (AssertionError,KeyError,TypeError,IndexError) as e:return dict(passed=False,errors=[repr(e)],scope='25-controller correctness HOLD')
