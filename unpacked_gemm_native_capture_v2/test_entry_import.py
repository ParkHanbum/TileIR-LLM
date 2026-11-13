"""CPU fresh-process actual --run-gpu entry import probe; never loads a backend.
Only preflight lock identity is simulated using /dev/null. No real lock access,
GPU query, torch/vllm import, compilation, or model is permitted.
"""
import builtins, hashlib, json, os, runpy, signal, sys
from pathlib import Path
from unittest.mock import patch
R=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();expect=sys.argv[3]
P=R.parent
assert not any(x.split('.')[0] in ('torch','vllm','cuda','triton') for x in sys.modules)
sys.path[:]=[str(R)]+[x for x in sys.path if x and Path(x).resolve()!=P and Path(x).resolve()!=R]
assert str(P) not in sys.path and 'PYTHONPATH' not in os.environ
LOCK=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock')
fd=os.open('/dev/null',os.O_RDONLY);fake=os.fstat(fd);old_stat=Path.stat
os.environ.update(VLLM_COMPARISON_PARENT_APPROVED='1',ALL_GEMM_LOCK_FD=str(fd),CUDA_VISIBLE_DEVICES='')
import cleanup
cleanup.final_cleanup=lambda *a,**k:None  # Test-only: no CUDA cleanup is applicable.
real_import=builtins.__import__;hit=[]
class BackendBoundary(BaseException):pass
def guarded(name,*args,**kw):
 if name.split('.')[0] in ('torch','vllm','cuda','triton'):
  assert name=='torch',('Unexpected backend import',name)
  assert 'packed_vllm_runtime_compare_v4.gates' in sys.modules
  assert callable(sys.modules['packed_vllm_runtime_compare_v4.gates'].compare_public)
  assert sys.modules['observer'].__file__==str(R/'observer.py')
  hit.append(name);raise BackendBoundary('Expected CPU-only stop before backend import')
 return real_import(name,*args,**kw)
sys.argv=[str(R/'run.py'),'--run-gpu','--output',str(output)]
status=None
try:
 with patch.object(Path,'stat',lambda p,*a,**k:fake if p==LOCK else old_stat(p,*a,**k)),patch.object(builtins,'__import__',guarded):
  runpy.run_path(str(R/'run.py'),run_name='__main__')
except ModuleNotFoundError as e:
 assert expect=='missing' and e.name=='packed_vllm_runtime_compare_v4' and not hit,(expect,repr(e),hit)
 status='reproduced_original_missing_sibling'
except BackendBoundary:
 assert expect=='resolved' and hit==['torch']
 status='actual_nondry_sibling_import_pass_before_backend'
finally:
 signal.alarm(0);os.close(fd)
assert status is not None
assert not any(x.split('.')[0] in ('torch','vllm','cuda','triton') for x in sys.modules)
print(json.dumps(dict(status=status,backend_modules_loaded=0,model_runs=0,gpu_queries=0,real_lock_operations=0,synthetic_preflight=True)))
