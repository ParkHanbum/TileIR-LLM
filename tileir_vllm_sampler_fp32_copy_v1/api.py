"""Fresh finite BF16-to-FP32 sampler copy. No autograd/native fallback."""
def validate_metadata(m):
 if m['shape']!=(32,248320) or m['stride']!=(248320,1) or m['offset']!=0 or m['dtype']!='torch.bfloat16':raise ValueError('actual contiguous BF16 sampler layout only')
 return m['shape']
def sampler_fp32_copy(x):
 import torch
 if not isinstance(x,torch.Tensor) or x.layout!=torch.strided or x.device.type!='cuda' or x.requires_grad or x.is_conj() or x.is_neg():raise ValueError('CUDA non-grad ordinary tensor required')
 shape=validate_metadata(dict(shape=tuple(x.shape),stride=tuple(x.stride()),offset=x.storage_offset(),dtype=str(x.dtype)))
 if not bool(torch.isfinite(x).all()):raise ValueError('nonfinite payloads not qualified; no launch')
 import cuda.tile as ct
 from .kernels import cast_kernel
 with torch.cuda.device(x.device):
  y=torch.empty(shape,dtype=torch.float32,device=x.device)
  n=x.numel();ct.launch(torch.cuda.current_stream(x.device),((n+1023)//1024,1,1),cast_kernel,(x.view(-1),y.view(-1),n,1024))
 return y
