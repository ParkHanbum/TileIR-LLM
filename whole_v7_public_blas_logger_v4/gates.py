from pathlib import Path
from whole_lean_profile_prepare_v4.admission import require_ready
from whole_lean_profile_prepare_v4.gates import audit as profile_audit
R=Path(__file__).resolve().parent
def qualification(root):
 root=Path(root);assert root.parent==R.parent and root.name in tuple(R.name+'_process'+str(i) for i in range(3))
 return require_ready()
def audit(result,root=None):
 base=profile_audit(result,root)
 if not base['passed']:return base
 try:
  for mode in ('native','lean'):
   d=result[mode+'_logger'];assert d['request_gate'] and not d['disable_errors']
   assert d['classic_api_counts'].get('cublasGemmEx',0)==(2448 if mode=='native' else 0)
   if mode=='lean':assert d['classic_api_counts']=={}
  return dict(base,public_api_log_audit=True)
 except (AssertionError,KeyError,TypeError):return dict(passed=False,scope='public API logger')
