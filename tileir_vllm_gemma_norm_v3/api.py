"""Narrow functional GemmaRMSNorm facade; not _C in-place fused norm ABI."""
import math,numbers

def validate(x,raw_weight,epsilon,residual=None,*,require_cuda=True):
 import torch
 if isinstance(epsilon,bool) or not isinstance(epsilon,numbers.Real) or not math.isfinite(float(epsilon)) or epsilon<=0:raise ValueError('epsilon must be positive finite real, not bool')
 for name,t in [('x',x),('raw_weight',raw_weight)]+([] if residual is None else [('residual',residual)]):
  if not isinstance(t,torch.Tensor):raise TypeError(name+' must be Tensor')
  if t.layout!=torch.strided or t.requires_grad:raise ValueError('strided inference tensors only')
  if t.device!=x.device or (require_cuda and t.device.type!='cuda'):raise ValueError('same CUDA device required')
  if t.storage_offset()<0:raise ValueError('negative storage offset')
 if x.dtype!=torch.bfloat16 or x.ndim!=2 or tuple(x.shape) not in ((32,2560),(1696,2560)):raise NotImplementedError('X scope BF16 M32/1696 K2560')
 if x.stride(1)!=1 or x.stride(0)<2560:raise ValueError('X inner contiguous, nonoverlapping rows required')
 if raw_weight.dtype not in (torch.bfloat16,torch.float32) or raw_weight.ndim!=1 or raw_weight.shape[0]!=2560 or raw_weight.stride(0)!=1:raise ValueError('raw weight BF16/FP32 vector2560 stride1')
 if residual is not None:
  if residual.dtype!=x.dtype or residual.shape!=x.shape or residual.stride(1)!=1 or residual.stride(0)<2560:raise ValueError('residual must match X with inner contiguous nonoverlapping rows')
 return x.shape[0],float(epsilon)

def _enqueue(x,w,epsilon,residual,out,residual_out,m):
 import torch
 import cuda.tile as ct
 from .kernels import square_kernel,mean_kernel,output_kernel
 # All temporaries allocated and used on the same current stream. Torch's
 # stream-aware allocator retains their storage until queued consumers finish.
 with torch.cuda.device(x.device):
  stream=torch.cuda.current_stream(x.device)
  square=torch.empty(x.shape,dtype=torch.float32,device=x.device)
  mean=torch.empty((m,),dtype=torch.float32,device=x.device)
  ct.launch(stream,(m,1,1),square_kernel,(x,x if residual is None else residual,square,out if residual_out is None else residual_out,residual is not None,4096))
  ct.launch(stream,(m,1,1),mean_kernel,(square,mean))
  ct.launch(stream,(m,1,1),output_kernel,(x,w,x if residual is None else residual,mean,out,epsilon,residual is not None,4096))

def gemma_rms_norm(x,raw_weight,epsilon,residual=None):
 """Fresh functional output(s); no input writes, vendor fallback or runtime patch.

 Raw learned weight, NOT pre-added effective weight. Weight promotion+1 and
 residual add/statistics use FP32; residual return and final normalized output
 BF16. No-residual returns Tensor; residual returns (Tensor,Tensor). The two
 outputs are fresh separate allocations. Current stream; no sync. Unqualified
 until actual vLLM captures/numerical comparison. Not an in-place API.
 """
 import torch
 m,eps=validate(x,raw_weight,epsilon,residual)
 out=torch.empty(x.shape,dtype=x.dtype,device=x.device)
 residual_out=None if residual is None else torch.empty(x.shape,dtype=x.dtype,device=x.device)
 _enqueue(x,raw_weight,eps,residual,out,residual_out,m)
 return out if residual is None else (out,residual_out)
