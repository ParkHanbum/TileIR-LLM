"""Narrow out-provided vLLM SiLU-and-mul companion, not a registered torch op."""
def validate(out, x, *, require_cuda=True):
    import torch
    for name,t in [('x',x),('out',out)]:
        if not isinstance(t,torch.Tensor): raise TypeError(name+' must be Tensor')
        if t.layout!=torch.strided or t.dtype!=torch.bfloat16 or t.ndim!=2:
            raise ValueError(name+' must be 2D strided BF16')
        if t.requires_grad or not t.is_contiguous(): raise ValueError('contiguous no-grad tensors required')
        if require_cuda and t.device.type!='cuda': raise ValueError('CUDA tensors required')
    if x.shape not in ((32,18432),(1696,18432)):
        raise NotImplementedError('only captured Qwen M32/1696, packed width18432')
    if out.shape!=(x.shape[0],9216) or out.device!=x.device:
        raise ValueError('out shape/device mismatch')
    # Conservative: reject even disjoint views of one allocation. CUDA source
    # uses restrict pointers; no in-place/overlap guarantee is being invented.
    os, xs = out.untyped_storage(), x.untyped_storage()
    ob, xb = os.data_ptr(), xs.data_ptr()
    if ob < xb + xs.nbytes() and xb < ob + os.nbytes():
        raise ValueError('out and x may not share storage: entire backing intervals must not overlap')
    return x.shape[0]

def _enqueue(out,x,m):
    import torch
    import cuda.tile as ct
    from .kernels import silu_mul_kernel
    with torch.cuda.device(x.device):
        ct.launch(torch.cuda.current_stream(x.device),(m,9,1),silu_mul_kernel,(x,out,1024))

def silu_and_mul(out,x):
    """Mutate supplied out; return None, like torch.ops._C.silu_and_mul.

    No allocation/copy/sync/fallback; validation precedes launch. Current stream.
    BF16 storage, FP32 exp/div, intermediate BF16 SiLU then FP32 multiply and
    BF16 output. Exact transcendental/rounding equivalence requires GPU tests.
    """
    m=validate(out,x)
    _enqueue(out,x,m)

def forward(x):
    """Allocating convenience, equivalent scope to SiluAndMul.forward_cuda."""
    import torch
    # Validate input before allocating; output metadata checked by public call.
    if not isinstance(x,torch.Tensor):raise TypeError('x must be Tensor')
    if (x.layout!=torch.strided or x.dtype!=torch.bfloat16 or x.device.type!='cuda'
        or x.requires_grad or not x.is_contiguous() or tuple(x.shape) not in ((32,18432),(1696,18432))):
        raise ValueError('expected supported contiguous CUDA BF16 packed input')
    out=torch.empty((x.shape[0],9216),dtype=x.dtype,device=x.device)
    silu_and_mul(out,x)
    return out
