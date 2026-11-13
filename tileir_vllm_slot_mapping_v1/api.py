"""Narrow CP1 public slot mapping companion; no registration or fallback."""
def validate(tables,indices,starts,positions,sizes,enabled,out,padded,*,cp_size=1,require_cuda=True):
 import torch
 if type(cp_size) is not int or cp_size!=1:raise NotImplementedError('CP_SIZE1 only')
 if not isinstance(tables,tuple) or not tables:raise ValueError('nonempty table tuple')
 if type(padded) is not int or padded<0:raise ValueError('padded nonnegative int')
 tensors=(*tables,indices,starts,positions,sizes,enabled,out)
 for t in tensors:
  if not isinstance(t,torch.Tensor) or t.layout!=torch.strided or not t.is_contiguous() or t.requires_grad:raise ValueError('contiguous strided integer/bool tensors only')
  if t.device!=out.device or (require_cuda and t.device.type!='cuda'):raise ValueError('same CUDA-visible device required')
 for t,dt in ((indices,torch.int64),(starts,torch.int32),(positions,torch.int64),(sizes,torch.int32),(enabled,torch.bool)):
  if t.ndim!=1 or t.dtype!=dt:raise ValueError('unsupported vector dtype/rank')
 if starts.numel()!=indices.numel()+1 or sizes.numel()!=len(tables) or enabled.numel()!=len(tables):raise ValueError('vector length mismatch')
 if out.ndim!=2 or out.dtype!=torch.int64 or out.shape[0]!=len(tables) or padded>out.shape[1]:raise ValueError('out int64 group/capacity mismatch')
 for t in tables:
  if t.ndim!=2 or t.dtype!=torch.int32 or min(t.shape)<=0:raise ValueError('tables int32 positive matrices')
 os=out.untyped_storage();ob=os.data_ptr()
 for t in tensors[:-1]:
  s=t.untyped_storage();b=s.data_ptr()
  if ob<b+s.nbytes() and b<ob+os.nbytes():raise ValueError('entire out backing overlaps input')
 return indices.numel()
def validate_values_cpu(tables,indices,starts,positions,sizes,enabled,out,padded):
 validate(tables,indices,starts,positions,sizes,enabled,out,padded,require_cuda=False)
 if any(t.device.type!='cpu' for t in (*tables,indices,starts,positions,sizes,enabled,out)):raise ValueError('CPU snapshots only')
 q=starts.tolist();ids=indices.tolist()
 if q[0]!=0 or any(b<a for a,b in zip(q,q[1:])) or q[-1]>padded or q[-1]>positions.numel():raise ValueError('invalid token prefix sum/capacity')
 if any(p<0 for p in positions[:q[-1]].tolist()):raise ValueError('nonnegative positions only')
 for g,t in enumerate(tables):
  bs=int(sizes[g]);en=bool(enabled[g])
  if bs<=0:raise ValueError('positive block size required')
  for r,req in enumerate(ids):
   if req<0:continue
   if req>=t.shape[0]:raise ValueError('request row out of range')
   for pos in positions[q[r]:q[r+1]].tolist():
    bi=pos//bs if en else 0
    if bi>=t.shape[1]:raise ValueError('block column out of range')
    # Narrow contract excludes overflow before promotion in native int32 product.
    block=int(t[req,bi])
    if block<0 or block*bs>2147483647:raise ValueError('nonnegative int32 block product domain')
def reference_cpu(tables,indices,starts,positions,sizes,enabled,out,padded):
 validate_values_cpu(tables,indices,starts,positions,sizes,enabled,out,padded)
 q=starts.tolist()
 for g,t in enumerate(tables):
  bs=int(sizes[g]);en=bool(enabled[g])
  for r,req in enumerate(indices.tolist()):
   for j in range(q[r],q[r+1]):
    pos=int(positions[j]);out[g,j]=int(t[req,pos//bs])*bs+pos%bs if en and req>=0 else -1
  out[g,q[-1]:]=-1
 return out[:,:padded]
def compute_slot_mappings(tables,indices,starts,positions,sizes,enabled,out,padded,*,cp_size=1):
 """Mutate full output capacity; return aliasing prefix, possibly noncontiguous.
 Data preconditions checked only by qualification CPU helper; runtime does not sync.
 Negative request IDs emit PAD(-1). Disabled groups emit PAD. CP>1 unsupported.
 """
 n=validate(tables,indices,starts,positions,sizes,enabled,out,padded,cp_size=cp_size)
 import torch,cuda.tile as ct
 from .kernels import slot_group
 with torch.cuda.device(out.device):
  stream=torch.cuda.current_stream(out.device)
  for g,t in enumerate(tables):ct.launch(stream,(n+1,1,1),slot_group,(t,indices,starts,positions,sizes,enabled,out,g,n,256))
 return out[:,:padded]
