"""Narrow independent functional GDN gated RMSNorm, not yet GPU-qualified."""
import math

def validate_metadata(x,z,w,epsilon):
 if type(epsilon) not in (float,int) or not math.isfinite(epsilon) or epsilon<=0:raise ValueError('positive finite epsilon required')
 for label,s in [('X',x),('Z',z)]:
  if tuple(s['shape']) not in ((1696,32,128),(32,32,128)) or s['dtype']!='torch.bfloat16':raise NotImplementedError(label+' expected BF16 T/32/128')
  a,b,c=s['stride']
  if c!=1 or b<128 or a<31*b+128 or s['offset']<0:raise ValueError(label+' unsupported overlapping/inner-strided layout')
 if x['shape']!=z['shape']:raise ValueError('X/Z shapes differ')
 if tuple(w['shape'])!=(128,) or tuple(w['stride'])!=(1,) or w['dtype'] not in ('torch.bfloat16','torch.float32') or w['offset']<0:raise ValueError('W raw vector128 BF16/FP32')
 return x['shape'][0],float(epsilon)

def validate(x,z,weight,epsilon=1e-6):
 import torch
 def meta(t):
  if not isinstance(t,torch.Tensor) or t.layout!=torch.strided or t.requires_grad or t.is_conj() or t.is_neg():raise ValueError('non-grad strided Tensor/Parameter required')
  if t.device.type!='cuda' or t.device!=x.device:raise ValueError('same CUDA device required')
  return dict(shape=tuple(t.shape),stride=tuple(t.stride()),offset=t.storage_offset(),dtype=str(t.dtype))
 return validate_metadata(meta(x),meta(z),meta(weight),epsilon)

def gated_rms_norm(x,z,weight,epsilon=1e-6):
 """norm_before_gate=True, silu, group_size=None, bias=None only.
 Fresh BF16 output; input preservation. Weight is raw multiplier, NOT Gemma+1.
 No vendor fallback, synchronization, runtime patch or backward support.
 """
 import torch
 tokens,eps=validate(x,z,weight,epsilon)
 out=torch.empty(x.shape,dtype=x.dtype,device=x.device)
 import cuda.tile as ct
 from .kernels import gated_norm_kernel
 with torch.cuda.device(x.device):ct.launch(torch.cuda.current_stream(x.device),(tokens*32,1,1),gated_norm_kernel,(x,z,weight,out,eps))
 return out
