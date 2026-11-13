"""Audit projections reuse proven numerical gates, without changing saved evidence."""
from pathlib import Path
from .adapter import ORDER
from .gemm_gate import audit as gemm_audit
from .sixteen_gate import audit as sixteen_audit
from sampler_three_ordered_runtime_integration_v2.result_gate import audit as top5_audit
R=Path(__file__).resolve().parent

def project(r,key):
 out=dict(r);out['adapter_restore']=[dict(r['adapter_restore'][0])]
 for arm in ('shadow_request','replacement_request'):
  a=dict(r[arm]);allparts=a['selected_dispatch']['components']
  if key=='sixteen':
   selected={k:v for k,v in allparts.items() if k not in ('all_gemm','sampler_three')}
   a['selected_dispatch']=dict(components=selected,shadow_pass=all(v['shadow_pass']for v in selected.values()),supported_calls=sum(v['supported_calls']for v in selected.values()),supported_vendor_fallback_calls=sum(v.get('supported_vendor_fallback_calls',0)for v in selected.values()))
  else:a['selected_dispatch']=allparts[key]
  out[arm]=a
 if key=='all_gemm':out['adapter_install']=[r['adapter_install'][0]['components'][key]]
 return out

def audit(r):
 try:
  assert r['complete'] and not r.get('error') and not r.get('cleanup_errors')
  c=r['cache_preseed'];assert c['destination']==str(R/'cache_v1') and c['files_verified'] and c['separate_inodes_verified']
  assert r['adapter_install'][0]['install_order']==list(ORDER)
  restore=r['adapter_restore'][0];assert restore['restored'] and restore['save_guards_restored'] and restore['close_order']==list(reversed(ORDER))
  for phase in ('shadow_request','replacement_request'):
   d=r[phase]['selected_dispatch'];assert set(d['components'])==set(ORDER) and not d['failed'] and d['shadow_pass']
   assert d['selected_boundary_calls']==5648 and d['selected_logical_operations']==5680 and d['selected_vendor_fallback_calls']==0 and d['installation_order']==list(ORDER)
  details=dict(sixteen=sixteen_audit(project(r,'sixteen')),gemm=gemm_audit(project(r,'all_gemm'),R),sampler_three=top5_audit(project(r,'sampler_three')))
  assert all(v['passed'] for v in details.values()),details
  return dict(passed=True,errors=[],components=details,scope='5680 selected logical operations; 5648 helper/GEMM boundaries; not GPU launches or full-model100% support')
 except (AssertionError,KeyError,TypeError,IndexError) as e:return dict(passed=False,errors=[repr(e)],scope='Composition correctness HOLD')
