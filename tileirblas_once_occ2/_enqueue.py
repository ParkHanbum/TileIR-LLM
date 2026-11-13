"""PRIVATE implementation. Only api.gemm produces the validated plan.

Not exported; callers must not invoke with arbitrary plans or mutated metadata.
No kernel math is duplicated. Public API never accepts a plan or path argument.
"""
from importlib import import_module
from functools import lru_cache


def _allocate(A,plan):
    import torch
    return torch.empty((plan.m,plan.n),dtype=torch.bfloat16,device=A.device)


def _resolve_kernel(module,name):
    return getattr(import_module(module),name)


@lru_cache(maxsize=1)
def _prefill_kernel_occ2():
    # Public API: distinct kernel/JIT cache, retained across calls. No compile here.
    return _resolve_kernel("tileirblas_prefill.kernels","prefill_kernel").replace_hints(occupancy=2)


def _launch_kernel(A,kernel,grid,args):
    import torch
    import cuda.tile as ct
    with torch.cuda.device(A.device):
        ct.launch(torch.cuda.current_stream(A.device),grid,kernel,args)


def enqueue(A,W,out,plan,path):
    if out is None:out=_allocate(A,plan)
    if path=='empty':return out
    m,n,k=plan.m,plan.n,plan.k
    if path=='decode64k64':
        module,name='tileirblas_decode.kernels','decode_kernel'
        grid=(1,(n+63)//64,1);args=(A,W,out,k,64,64)
    elif path=='prefill128':
        module,name='tileirblas_prefill.kernels','prefill_kernel'
        grid=((m+127)//128,(n+127)//128,1);args=(A,W,out,k,32)
    elif path=='residue':
        module,name='tileirblas_blocked.fallback','nt_kernel'
        grid=((m+255)//256,(n+127)//128,1);args=(A,W,out,m,n,k)
    elif path=='blocked':
        module,name='tileirblas_blocked.kernels','blocked_kernel'
        bm,bn=(32,128) if m<=32 else ((32,32) if n<=32 else (256,128))
        grid=((m+bm-1)//bm,(n+bn-1)//bn,1);args=(A,W,out,m,n,k,bm,bn)
    else:
        raise RuntimeError('invalid internal path')
    kernel=_prefill_kernel_occ2() if path=="prefill128" else _resolve_kernel(module,name)
    _launch_kernel(A,kernel,grid,args)
    return out
