from .contract import vector
def selected_1d_checked(x,ids):
 import torch
 if not isinstance(x,torch.Tensor) or not x.is_cuda or x.layout!=torch.strided or x.requires_grad:raise ValueError('Readonly CUDA vector required')
 size,itemsize=vector(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.untyped_storage().nbytes(),x.is_neg(),x.is_conj())
 if not isinstance(ids,torch.Tensor) or ids.device!=x.device or ids.dtype!=torch.int64 or ids.layout!=torch.strided or ids.requires_grad or ids.is_neg() or ids.is_conj() or ids.ndim!=1 or ids.stride()!=(1,) or not 0<=ids.numel()<=1024:raise ValueError('Readonly int64 indices required')
 from public_i64_range_guard_v1.api import all_in_range_i64
 # Repeated bounds guard is intentional; no dependent read before its success.
 if not all_in_range_i64(ids,0,size):raise ValueError('Selected index out of bounds')
 n=ids.numel();out=torch.empty((n,),dtype=x.dtype,device=x.device)
 if n:
  import cuda.tile as ct
  from .kernel import gather_raw
  bits=torch.uint32 if itemsize==4 else torch.uint16
  ct.launch(torch.cuda.current_stream(x.device),(1,),gather_raw,(x.view(bits),ids,out.view(bits),n,1<<(n-1).bit_length()))
 return out

def selected_all_zero(x,ids):
 # Independent fresh raw gather then existing typed EQ0. Do not fold equality
 # into the gather or eliminate the original host assertion ordering.
 import torch
 # Existing EQ0 supports FP32 only: reject before gather/allocation.
 if not isinstance(x,torch.Tensor) or x.dtype!=torch.float32:raise ValueError('FP32 EQ0 contract required')
 selected=selected_1d_checked(x,ids)
 from public_eq_all_guard_v2.api import eq_all
 return eq_all(selected,0)
