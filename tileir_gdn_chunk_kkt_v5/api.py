"""Experimental KKT only; explicit BK, no autotune winner or raw-equality claim."""
def chunk_scaled_dot_kkt(k,g,beta,cu_seqlens,chunk_indices,*,bk):
 assert type(bk) is int and bk==64
 import torch
 from .contract import descriptor
 from tileir_gdn_chunk_cumsum_v1.contract import validate_metadata
 shape=descriptor(k.shape,k.stride(),g.shape,g.stride(),beta.shape,beta.stride(),bk)
 assert k.dtype==torch.bfloat16 and g.dtype==beta.dtype==torch.float32
 assert all(t.is_cuda and t.device==k.device and t.storage_offset()==0 and not t.requires_grad for t in (k,g,beta,cu_seqlens,chunk_indices))
 assert tuple(cu_seqlens.shape)==(33,) and tuple(chunk_indices.shape)==(32,2)
 assert cu_seqlens.dtype==chunk_indices.dtype==torch.int32 and cu_seqlens.is_contiguous() and chunk_indices.is_contiguous()
 validate_metadata(cu_seqlens.tolist(),chunk_indices.tolist())
 import cuda.tile as ct
 from .kernels import kkt
 out=torch.empty(shape,dtype=torch.float32,device=k.device)
 ct.launch(torch.cuda.current_stream(k.device),(32,32),kkt,(k,g,beta,cu_seqlens,chunk_indices,out,bk))
 return out
