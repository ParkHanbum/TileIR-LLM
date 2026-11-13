"""Narrow public-source indexed BF16 width4 decode; correctness-first."""
def validate_metadata(x,state,w,idx,slots,null_block_id=0):
 if type(null_block_id) is not int or null_block_id!=0:raise ValueError('observed null block 0 only')
 if tuple(x['shape'])!=(32,8192) or x['dtype']!='torch.bfloat16':raise ValueError('X BF16 [32,8192]')
 if x['stride'][1]!=1 or x['stride'][0]<8192:raise ValueError('X disjoint rows required')
 if len(state['shape'])!=3 or state['shape'][1:]!=(8192,3) or state['dtype']!='torch.bfloat16':raise ValueError('state BF16 [capacity,8192,3]')
 if state['stride'][1]!=1 or state['stride'][2]<8192 or state['stride'][0]<2*state['stride'][2]+8192:raise ValueError('state DS logical nonoverlap required')
 if w['shape']!=(8192,4) or w['stride']!=(4,1) or w['dtype']!='torch.bfloat16':raise ValueError('weight BF16 contiguous [8192,4]')
 if idx['shape']!=(32,) or idx['stride']!=(1,) or idx['dtype']!='torch.int32':raise ValueError('indices int32 contiguous [32]')
 if any(type(a['offset']) is not int or a['offset']<0 for a in (x,state,w,idx)):raise ValueError('offset')
 if len(slots)!=32 or any(type(s) is not int or s<0 or s>=state['shape'][0] for s in slots):raise ValueError('slot bounds')
 valid=[s for s in slots if s!=null_block_id]
 if len(set(valid))!=len(valid):raise ValueError('duplicate valid slots')
 return len(valid)

def causal_conv1d_update_indexed(x,conv_state,weight,conv_state_indices,*,activation='silu',null_block_id=0):
 """Mutates X and selected state, returns distinct same-storage X view.
 Bias=None/out=None/ordinary one-token BF16 only. No vendor fallback.
 CPU slot validation synchronizes deliberately; not performance qualified.
 """
 import torch
 ts=(x,conv_state,weight,conv_state_indices)
 def meta(t):
  if not isinstance(t,torch.Tensor) or t.layout!=torch.strided or t.requires_grad or t.is_conj() or t.is_neg():raise ValueError('strided non-grad tensor')
  if t.device.type!='cuda' or t.device!=x.device:raise ValueError('same CUDA device')
  return dict(shape=tuple(t.shape),stride=tuple(t.stride()),offset=t.storage_offset(),dtype=str(t.dtype))
 ms=[meta(t) for t in ts]
 if activation not in ('silu','swish'):raise ValueError('SiLU only')
 intervals=[(t.untyped_storage().data_ptr(),t.untyped_storage().data_ptr()+t.untyped_storage().nbytes()) for t in ts]
 for i,a in enumerate(intervals):
  for b in intervals[i+1:]:
   if max(a[0],b[0])<min(a[1],b[1]):raise ValueError('input owner storages must not overlap')
 validate_metadata(*ms,conv_state_indices.detach().cpu().tolist(),null_block_id)
 import cuda.tile as ct
 from .kernels import update_kernel
 with torch.cuda.device(x.device):ct.launch(torch.cuda.current_stream(x.device),(32,32,1),update_kernel,(x,conv_state,weight,conv_state_indices))
 return x.view_as(x)
