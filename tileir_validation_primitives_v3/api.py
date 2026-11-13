"""Uncompiled source-grounded candidate; not installed into any runtime."""
def all_finite(x):
 import torch
 from .contract import descriptor,status_boolean
 assert isinstance(x,torch.Tensor) and x.device.type=='cuda'
 n=descriptor(tuple(x.shape),str(x.dtype),x.storage_offset(),x.is_contiguous())
 if n==0:return True # same all(empty) identity; not a model-supported empty input
 import cuda.tile as ct
 from .kernels import count_nonfinite,reduce_counts
 blocks=(n+1023)//1024;width=1<<(blocks-1).bit_length()
 counts=torch.empty((blocks,),dtype=torch.int32,device=x.device)
 status=torch.empty((1,),dtype=torch.int32,device=x.device)
 flat=torch.empty(0,dtype=x.dtype,device=x.device).set_(x.untyped_storage(),x.storage_offset(),(n,),(1,))
 assert not flat.is_neg() and not flat.is_conj() and flat.data_ptr()==x.data_ptr()
 stream=torch.cuda.current_stream(x.device)
 ct.launch(stream,(blocks,),count_nonfinite,(flat,counts,n))
 ct.launch(stream,(1,),reduce_counts,(counts,status,width))
 return status_boolean(int(status.item()),n)
def selected_predicate(x):
 import torch
 if x.dtype in (torch.bfloat16,torch.float32):return all_finite(x)
 # Preserve original integer metadata predicate, ordering and synchronization.
 return bool(torch.isfinite(x).all())
