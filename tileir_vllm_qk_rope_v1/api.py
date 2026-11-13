"""Observed BF16 Qwen QK norm/partial MRoPE/gate functional facade. No fallback."""
import math
def validate_metadata(q,k,qw,kw,c,p,eps,nq,nk,d,rot,sections,beta):
 if (nq,nk,d,rot)!=(16,4,256,64) or tuple(sections or ())!=(11,11,10) or eps!=1e-6 or beta!=1.0:raise NotImplementedError('Observed Qwen constants only')
 M=q['shape'][0]
 if M not in (1696,32):raise NotImplementedError('Observed token counts only')
 for x,shape in [(q,(M,8192)),(k,(M,1024)),(qw,(256,)),(kw,(256,)),(c,(262144,64))]:
  if tuple(x['shape'])!=shape or x['dtype']!='torch.bfloat16' or x['stride'][-1]!=1 or x['offset']<0:raise ValueError('BF16 shape/inner stride/offset contract')
  if len(shape)==2 and x['stride'][0]<shape[1]:raise ValueError('Overlapping rows')
 if tuple(p['shape'])!=(3,M) or p['dtype']!='torch.int64' or p['stride'][1]!=1 or p['stride'][0]<M or p['offset']<0:raise ValueError('Position layout contract')
 return M

def fused_qk_rmsnorm_rope_gate(q_gate,k,q_weight,k_weight,cos_sin_cache,positions,eps,num_q_heads,num_kv_heads,head_dim,rotary_dim,mrope_section=None,norm_beta=0.0):
 import torch
 vals=(q_gate,k,q_weight,k_weight,cos_sin_cache,positions)
 def meta(t):
  if not isinstance(t,torch.Tensor) or t.layout!=torch.strided or t.requires_grad or t.is_conj() or t.is_neg() or t.device.type!='cuda' or t.device!=q_gate.device:raise ValueError('Same-device inference strided tensors required; detach raw weights')
  return dict(shape=tuple(t.shape),stride=tuple(t.stride()),offset=t.storage_offset(),dtype=str(t.dtype))
 M=validate_metadata(*(meta(t) for t in vals),eps,num_q_heads,num_kv_heads,head_dim,rotary_dim,mrope_section,norm_beta)
 q=torch.empty((M,4096),dtype=q_gate.dtype,device=q_gate.device);ko=torch.empty((M,1024),dtype=k.dtype,device=k.device);g=torch.empty_like(q)
 import cuda.tile as ct
 from .kernels import kernel
 with torch.cuda.device(q_gate.device):ct.launch(torch.cuda.current_stream(q_gate.device),(M,20,1),kernel,(*vals,q,ko,g))
 return q,ko,g
