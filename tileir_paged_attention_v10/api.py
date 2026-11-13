"""Narrow mathematical attention candidate; native raw equivalence unqualified."""
from .contract import validate_values
def attention(q,k,v,cuq,lens,table,out,*,scale=0.0625,causal=True):
 import torch
 from tileir_paged_attention_v9.kernels import kernel
 from .kernels import two_block_kernel
 import cuda.tile as ct
 ts=(q,k,v,cuq,lens,table,out)
 assert all(isinstance(t,torch.Tensor) and t.device==q.device and t.is_cuda and not t.requires_grad for t in ts)
 assert scale==0.0625 and causal is True
 assert q.dtype==k.dtype==v.dtype==out.dtype==torch.bfloat16 and tuple(q.shape[1:])==(16,256) and out.shape==q.shape
 assert q.stride()==out.stride()==(4096,256,1) and q.storage_offset()==out.storage_offset()==0
 assert k.shape==v.shape and tuple(k.shape[1:])==(528,4,256) and k.stride()==v.stride()==(1081344,2048,512,1)
 assert k.untyped_storage().data_ptr()==v.untyped_storage().data_ptr() and v.storage_offset()==k.storage_offset()+256
 assert cuq.dtype==lens.dtype==table.dtype==torch.int32 and cuq.stride()==lens.stride()==(1,) and table.stride()==(8,1) and tuple(table.shape)==(32,8)
 width=validate_values(cuq.cpu().tolist(),lens.cpu().tolist(),table.cpu().tolist(),k.shape[0],k.shape[1]);assert q.shape[0]==32*width
 a=out.untyped_storage();lo,hi=a.data_ptr(),a.data_ptr()+a.nbytes()
 for t in ts[:-1]:
  s=t.untyped_storage();assert not(lo<s.data_ptr()+s.nbytes() and s.data_ptr()<hi)
 selected=two_block_kernel if width==1 and int(lens[0].item())>64 else kernel
 ct.launch(torch.cuda.current_stream(q.device),(q.shape[0],16),selected,(q,k,v,lens,table,out,width))
 return out
