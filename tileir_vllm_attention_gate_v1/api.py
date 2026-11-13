"""Independent inference-only functional facade; no backward or fallback."""
def validate_metadata(x,g):
 for m in (x,g):
  if tuple(m['shape']) not in ((1696,4096),(32,4096)) or m['dtype']!='torch.bfloat16':raise ValueError('Expected actual TP1 BF16 Mx4096')
  if tuple(m['stride'])[1]!=1 or m['stride'][0]<4096 or m['offset']<0:raise ValueError('Unsupported layout')
 if x['shape']!=g['shape']:raise ValueError('Shapes differ')
 return x['shape'][0]
def attention_sigmoid_gate(x,gate):
 import torch
 def meta(t):
  if not isinstance(t,torch.Tensor) or t.layout!=torch.strided or t.requires_grad or t.is_conj() or t.is_neg() or t.device.type!='cuda' or t.device!=x.device:raise ValueError('Same CUDA non-grad strided inference tensors required')
  return dict(shape=tuple(t.shape),stride=tuple(t.stride()),offset=t.storage_offset(),dtype=str(t.dtype))
 m=validate_metadata(meta(x),meta(gate));out=torch.empty((m,4096),device=x.device,dtype=x.dtype)
 import cuda.tile as ct
 from .kernels import kernel
 with torch.cuda.device(x.device):ct.launch(torch.cuda.current_stream(x.device),(m,1,1),kernel,(x,gate,out))
 return out
