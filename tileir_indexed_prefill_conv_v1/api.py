"""Fixed actual ordinary prefill functional adapter, not full native ABI."""
def validate_metadata(x,state,w,q,idx,initial,starts,slots,flags):
 if x['shape']!=(8192,1696) or x['dtype']!='torch.bfloat16' or x['stride'][0]!=1 or x['stride'][1]<8192:raise ValueError('X BF16 channel-last packed 32x53')
 if len(state['shape'])!=3 or state['shape'][1:]!=(8192,3) or state['dtype']!='torch.bfloat16':raise ValueError('state BF16 history3')
 if state['stride'][1]!=1 or state['stride'][2]<8192 or state['stride'][0]<2*state['stride'][2]+8192:raise ValueError('state logical cells must not overlap')
 if w['shape']!=(8192,4) or w['stride']!=(4,1) or w['dtype']!='torch.bfloat16':raise ValueError('weight BF16 contiguous 8192x4')
 for m,n,dtype in ((q,33,'torch.int32'),(idx,32,'torch.int32'),(initial,32,'torch.bool')):
  if m['shape']!=(n,) or m['stride']!=(1,) or m['dtype']!=dtype:raise ValueError('metadata layout/dtype')
 if any(type(m['offset']) is not int or m['offset']<0 for m in (x,state,w,q,idx,initial)):raise ValueError('offset')
 if starts!=list(range(0,1697,53)) or any(type(v)is not int for v in starts):raise ValueError('only exact 32 sequences of length53')
 if len(flags)!=32 or any(type(v)is not bool or v for v in flags):raise ValueError('has_initial_state False only')
 if len(slots)!=32 or any(type(v)is not int or not 0<v<state['shape'][0] for v in slots) or len(set(slots))!=32:raise ValueError('unique positive non-null slots only')
 return True

def causal_conv1d_prefill_indexed(x,weight,conv_states,query_start_loc,cache_indices,has_initial_state,*,activation='silu'):
 """Fresh channel-last BF16 output; update selected state with final3 input tokens.
 Only observed no-bias/non-null/no-initial-state32x53 scope. No fallback.
 Host metadata validation synchronizes deliberately; performance unqualified.
 """
 import torch
 ts=(x,conv_states,weight,query_start_loc,cache_indices,has_initial_state)
 def meta(t):
  if not isinstance(t,torch.Tensor) or t.layout!=torch.strided or t.requires_grad or t.is_conj() or t.is_neg():raise ValueError('non-grad strided tensors only')
  if t.device.type!='cuda' or t.device!=x.device:raise ValueError('same CUDA device')
  return dict(shape=tuple(t.shape),stride=tuple(t.stride()),offset=t.storage_offset(),dtype=str(t.dtype))
 ms=[meta(t) for t in ts]
 if activation not in ('silu','swish'):raise ValueError('SiLU only')
 intervals=[(t.untyped_storage().data_ptr(),t.untyped_storage().data_ptr()+t.untyped_storage().nbytes()) for t in ts]
 for i,a in enumerate(intervals):
  for b in intervals[i+1:]:
   if max(a[0],b[0])<min(a[1],b[1]):raise ValueError('separate input owner storages required')
 validate_metadata(*ms,query_start_loc.detach().cpu().tolist(),cache_indices.detach().cpu().tolist(),has_initial_state.detach().cpu().tolist())
 import cuda.tile as ct
 from .kernels import prefill_kernel
 with torch.cuda.device(x.device):
  y=torch.empty((1696,8192),device=x.device,dtype=x.dtype).transpose(0,1)
  ct.launch(torch.cuda.current_stream(x.device),(1696,32,1),prefill_kernel,(x,conv_states,weight,cache_indices,y))
 return y
