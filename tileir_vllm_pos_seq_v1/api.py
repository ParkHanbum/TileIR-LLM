"""CPU draft of public vLLM prepare_pos_seq_lens in-place ABI."""
def reference_checked(mapping,starts,computed,pos,seq):
 """CPU value-domain oracle; invalid-domain rejection is NOT GPU runtime validation."""
 n=len(mapping)
 if not 0<=n<=len(seq)<=32 or len(starts)<n+1:raise ValueError('capacities')
 if any(type(x) is not int for a in (mapping,starts,computed,pos,seq) for x in a):raise TypeError('integer vectors')
 if any(i<0 or i>=len(computed) for i in mapping):raise ValueError('state index')
 if starts[0]<0 or starts[n]>len(pos) or any(starts[i]>starts[i+1] for i in range(n)):raise ValueError('prefix')
 out=list(pos);lens=list(seq)
 for i,idx in enumerate(mapping):
  count=computed[idx];length=starts[i+1]-starts[i]
  if count<0 or count+length>2147483647:raise ValueError('nonnegative int32 range')
  lens[i]=count+length
  for j in range(length):out[starts[i]+j]=count+j
 for i in range(n,len(seq)):lens[i]=0
 return out,lens

def prepare_pos_seq_lens(idx_mapping,query_start_loc,num_computed_tokens,pos,seq_lens):
 """Mutates pos live query ranges and all seq_lens; returns None.
 Caller must establish value-domain preconditions using captured/owned metadata.
 No hidden host reads or fallback. GPU implementation not yet compiled/qualified.
 """
 import torch
 tensors=(idx_mapping,query_start_loc,num_computed_tokens,pos,seq_lens)
 dtypes=(torch.int64,torch.int32,torch.int32,torch.int64,torch.int32)
 for t,d in zip(tensors,dtypes):
  if not isinstance(t,torch.Tensor) or t.dtype!=d or t.ndim!=1 or t.stride(0)!=1 or t.requires_grad or t.device.type!='cuda' or t.device!=pos.device:raise ValueError('same CUDA contiguous integer vectors')
 n=idx_mapping.numel();capacity=seq_lens.numel()
 if not 0<=n<=capacity<=32 or capacity==0 or query_start_loc.numel()<n+1:raise ValueError('capacity scope')
 for output_index in (3,4):
  out=tensors[output_index]
  for input_index,other in enumerate(tensors):
   if output_index==input_index:continue
   a=out.data_ptr();b=other.data_ptr()
   if max(a,b)<min(a+out.numel()*out.element_size(),b+other.numel()*other.element_size()):raise ValueError('overlapping output/input storage')
 import cuda.tile as ct
 from .kernels import kernel
 with torch.cuda.device(pos.device):ct.launch(torch.cuda.current_stream(pos.device),(capacity,1,1),kernel,(*tensors,n))
