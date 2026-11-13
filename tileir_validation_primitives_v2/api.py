"""Experimental synchronous pre-write finite guard, CPU-prepared/uncompiled."""
def require_finite(x):
 import torch
 from .contract import validate_finite_descriptor,checked_status
 assert isinstance(x,torch.Tensor) and x.device.type=='cuda'
 n=validate_finite_descriptor(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.requires_grad,x.is_conj(),x.is_neg())
 import cuda.tile as ct
 from .kernels import count_nonfinite,reduce_counts
 blocks=(n+1023)//1024;reduce_width=1<<(blocks-1).bit_length()
 counts=torch.empty((blocks,),device=x.device,dtype=torch.int32)
 status=torch.empty((1,),device=x.device,dtype=torch.int32)
 stream=torch.cuda.current_stream(x.device)
 # Public metadata-only storage alias drops view flags, not data. Real sign/conjugation preserve finiteness.
 flat=torch.empty(0,dtype=x.dtype,device=x.device).set_(x.untyped_storage(),x.storage_offset(),(n,),(1,))
 assert not flat.is_conj() and not flat.is_neg() and flat.data_ptr()==x.data_ptr()
 ct.launch(stream,(blocks,),count_nonfinite,(flat,counts,n))
 ct.launch(stream,(1,),reduce_counts,(counts,status,reduce_width))
 # Host observes only one int32 status; synchronization remains REQUIRED before caller writes.
 checked_status(int(status.item()))
 return None
