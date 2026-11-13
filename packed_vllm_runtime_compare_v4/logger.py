"""Official public cuBLAS runtime logger; no loading a new vendor instance."""
import ctypes,hashlib,os
from pathlib import Path
LIB=Path('/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/nvidia/cu13/lib/libcublas.so.13')
LIB_SHA='25cde432108543626df85d887c730bf2cab96803ca16f9654e4abc8d2b45833e'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snapshot(p):
 p=Path(p);return {'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else 0,'sha256':digest(p) if p.exists() else None}
class PublicLogger:
 def __init__(self,path):
  self.path=Path(path);assert not self.path.exists()
  assert not any(k.startswith(('CUBLAS_LOG','CUBLASLT_LOG')) for k in os.environ)
  paths={Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines() if '/libcublas.so.' in line}
  assert paths=={LIB.resolve()},('Unexpected mapped cuBLAS',paths)
  assert digest(LIB)==LIB_SHA
  self.lib=ctypes.CDLL(str(LIB.resolve()),mode=os.RTLD_NOLOAD|os.RTLD_LOCAL)
  self.configure=self.lib.cublasLoggerConfigure
  self.configure.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_char_p];self.configure.restype=ctypes.c_int
  self.record={'library':str(LIB.resolve()),'library_sha256':LIB_SHA,'scope':'Untimed baseline semantic gate only; payload availability unknown until log inspection','calls':[]}
 def call(self,on):
  status=int(self.configure(on,0,0,os.fsencode(self.path) if on else None));self.record['calls'].append({'on':on,'status':status});assert status==0
 def __enter__(self):
  try:self.call(1)
  except BaseException:
   self.call(0);raise
  return self
 def __exit__(self,*exc):
  self.call(0);self.record['disabled_success']=True
  self.record['after_disable']=snapshot(self.path)
 def unchanged(self):
  now=snapshot(self.path);assert now==self.record['after_disable'],'Logger output changed after disable'
  self.record['off_verified_no_file_change']=True;return now
