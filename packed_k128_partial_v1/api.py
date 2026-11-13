"""Data-independent narrow shape dispatch; no setting changes or vendor fallback."""
TARGET=(32,64,2560)
def selected_path(m,n,k):
 if any(type(x) is not int for x in (m,n,k)):raise TypeError('Integer dimensions required')
 return 'bf16_k128' if (m,n,k)==TARGET else 'unchanged_base'

def tile_ranges(k):
 if type(k) is not int or k<=0 or k%128:raise ValueError('K must be a positive multiple of128')
 return tuple(range(p*4,(p+1)*4) for p in range(k//128))

def _validate(x,w,out):
 from tileirblas_blocked.api import validate_gemm
 return validate_gemm(x,w,out)

def _base(x,w,out):
 from tileirblas_once_occ2 import linear as original
 return original(x,w,out=out)

def _allocate(x,m,n):
 import torch
 return torch.empty((m,n),device=x.device,dtype=torch.bfloat16)

def _enqueue(x,w,y):
 import torch
 import cuda.tile as ct
 from .kernels import k128_bf16_partials
 with torch.cuda.device(x.device):
  ct.launch(torch.cuda.current_stream(x.device),(1,1,1),k128_bf16_partials,(x,w,y,2560,32,128))

def linear(x,w,out=None):
 plan=_validate(x,w,out)
 if selected_path(plan.m,plan.n,plan.k)=='unchanged_base':return _base(x,w,out)
 y=out if out is not None else _allocate(x,plan.m,plan.n)
 _enqueue(x,w,y)
 return y
