"""Public C wrapper adapter copied from reviewed reference v2; no GPU action on import."""
import ctypes
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent/'reference'
BLAS=Path('/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/nvidia/cu13/lib/libcublas.so.13')
class Reference:
 def __init__(self,torch):
  self.torch=torch; self.lib=ctypes.CDLL(str(ROOT/'libpublic_cublas.so'))
  P=ctypes.c_void_p; I=ctypes.c_int
  self.lib.ref_create.argtypes=[ctypes.POINTER(P)]; self.lib.ref_create.restype=I
  self.lib.ref_destroy.argtypes=[P];self.lib.ref_destroy.restype=I
  self.lib.ref_version.argtypes=[P,ctypes.POINTER(I)];self.lib.ref_version.restype=I
  self.lib.ref_bf16_nt.argtypes=[P]*5+[I]*6;self.lib.ref_bf16_nt.restype=I
  mapped={Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines() if '/libcublas.so.' in line}
  assert mapped=={BLAS.resolve()}, ('Unexpected loaded cuBLAS library',mapped)
  self.h=P();self.check(self.lib.ref_create(ctypes.byref(self.h)))
  try:
   v=I();self.check(self.lib.ref_version(self.h,ctypes.byref(v)));self.version=v.value
  except BaseException:
   self.close();raise
 def check(self,s):
  if s:raise RuntimeError('cuBLAS status '+str(s))
 def __call__(self,a,w,*,out):
  t=self.torch
  assert a.dtype==w.dtype==out.dtype==t.bfloat16 and a.device==w.device==out.device
  assert a.is_cuda and a.ndim==w.ndim==out.ndim==2
  m,k=a.shape;n=w.shape[0]
  assert w.shape[1]==k and out.shape==(m,n) and all(x.stride(1)==1 for x in (a,w,out))
  assert all(0 < x <= 2147483647 for x in (m,n,k,a.stride(0),w.stride(0),out.stride(0)))
  assert out.untyped_storage().data_ptr() not in (a.untyped_storage().data_ptr(),w.untyped_storage().data_ptr()), 'Output storage aliases input'
  self.check(self.lib.ref_bf16_nt(self.h,t.cuda.current_stream(a.device).cuda_stream,
   a.data_ptr(),w.data_ptr(),out.data_ptr(),m,n,k,a.stride(0),w.stride(0),out.stride(0)))
 def close(self):
  if self.h:self.check(self.lib.ref_destroy(self.h));self.h=None

def tensor_sha(t):
 return hashlib.sha256(t.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()
