def chunk_output(q,k,v,h,g,cu_seqlens,chunk_indices):
 import torch
 import cuda.tile as ct
 from .kernels import output
 from tileir_gdn_chunk_cumsum_v1.contract import validate_metadata
 ts=(q,k,v,h,g,cu_seqlens,chunk_indices)
 specs=[((1,1696,16,128),torch.bfloat16),((1,1696,16,128),torch.bfloat16),((1,1696,32,128),torch.bfloat16),((1,32,32,128,128),torch.bfloat16),((1,1696,32),torch.float32),((33,),torch.int32),((32,2),torch.int32)]
 assert all(isinstance(t,torch.Tensor) and t.is_cuda and t.device==q.device and not t.requires_grad and t.storage_offset()==0 and t.is_contiguous() and tuple(t.shape)==s and t.dtype==d for t,(s,d) in zip(ts,specs))
 validate_metadata(cu_seqlens.cpu().tolist(),chunk_indices.cpu().tolist())
 out=torch.empty_like(v)
 ct.launch(torch.cuda.current_stream(q.device),(32,32,4),output,(q,k,v,h,g,out))
 return out
