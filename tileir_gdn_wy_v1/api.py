def recompute_w_u_fwd(k,v,beta,g_cumsum,A,cu_seqlens,chunk_indices):
 import torch
 import cuda.tile as ct
 from .kernels import wy
 from tileir_gdn_chunk_cumsum_v1.contract import validate_metadata
 ts=(k,v,beta,g_cumsum,A,cu_seqlens,chunk_indices)
 assert all(isinstance(t,torch.Tensor) and t.is_cuda and t.device==k.device and not t.requires_grad and t.storage_offset()==0 and t.is_contiguous() for t in ts)
 specs=[((1,1696,16,128),torch.bfloat16),((1,1696,32,128),torch.bfloat16),((1,1696,32),torch.float32),((1,1696,32),torch.float32),((1,1696,32,64),torch.bfloat16),((33,),torch.int32),((32,2),torch.int32)]
 assert all(tuple(t.shape)==s and t.dtype==d for t,(s,d) in zip(ts,specs))
 validate_metadata(cu_seqlens.cpu().tolist(),chunk_indices.cpu().tolist())
 w=torch.empty_like(v);u=torch.empty_like(v)
 ct.launch(torch.cuda.current_stream(k.device),(32,32,2),wy,(k,v,beta,g_cumsum,A,w,u))
 return w,u
