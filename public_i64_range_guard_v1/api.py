from .contract import descriptor

def all_in_range_i64(x,lower,upper_exclusive):
 import torch
 if not isinstance(x,torch.Tensor) or x.device.type!='cuda' or x.layout!=torch.strided or x.requires_grad:raise ValueError('Readonly CUDA strided tensor required')
 n=descriptor(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.untyped_storage().nbytes(),lower,upper_exclusive,x.is_neg(),x.is_conj())
 if n==0:return True
 import cuda.tile as ct
 from .kernel import range_status
 flat=x.as_strided((n,),(1,),x.storage_offset())
 status=torch.empty((1,),dtype=torch.int32,device=x.device)
 assert status.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()
 stream=torch.cuda.current_stream(x.device)
 ct.launch(stream,(1,),range_status,(flat,status,n,1<<(n-1).bit_length(),lower,upper_exclusive))
 count=int(status.item())
 if not 0<=count<=n:raise RuntimeError('Invalid predicate status')
 return count==0
