def chunk_state(k,w,u,g,initial_state,cu_seqlens,chunk_indices,chunk_offsets):
 import torch
 import cuda.tile as ct
 from .kernels import state
 from tileir_gdn_chunk_cumsum_v1.contract import validate_metadata
 ts=(k,w,u,g,initial_state,cu_seqlens,chunk_indices,chunk_offsets)
 specs=[((1,1696,16,128),torch.bfloat16),((1,1696,32,128),torch.bfloat16),((1,1696,32,128),torch.bfloat16),((1,1696,32),torch.float32),((32,32,128,128),torch.float32),((33,),torch.int32),((32,2),torch.int32),((33,),torch.int64)]
 assert all(isinstance(t,torch.Tensor) and t.is_cuda and t.device==k.device and not t.requires_grad and t.storage_offset()==0 and t.is_contiguous() and tuple(t.shape)==s and t.dtype==d for t,(s,d) in zip(ts,specs))
 validate_metadata(cu_seqlens.cpu().tolist(),chunk_indices.cpu().tolist())
 assert chunk_offsets.cpu().tolist()==list(range(33))
 h=torch.empty((1,32,32,128,128),dtype=torch.bfloat16,device=k.device);vn=torch.empty_like(u);final=torch.empty_like(initial_state)
 ct.launch(torch.cuda.current_stream(k.device),(32,32,4),state,(k,w,u,g,initial_state,h,vn,final))
 return h,vn,final
