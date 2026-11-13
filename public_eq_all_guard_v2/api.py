from public_eq_all_guard_v2.contract import descriptor

def eq_all(x,constant):
 import torch
 if not isinstance(x,torch.Tensor) or not x.is_cuda or x.requires_grad:raise ValueError('CUDA typed readonly tensor required')
 d=descriptor(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.untyped_storage().nbytes(),constant,x.is_neg(),x.is_conj())
 if d['physical']==0:return True
 # Typed alias only: retain original storage offset, no address-to-tensor construction.
 flat=x.as_strided((d['physical'],),(1,))
 import cuda.tile as ct
 from public_eq_all_guard_v2.kernel import partial_eq,finish_eq
 n=d['physical'];groups=(n+1023)//1024;width=1<<(groups-1).bit_length()
 partial=torch.empty((groups,),device=x.device,dtype=torch.int32)
 out=torch.empty((1,),device=x.device,dtype=torch.int32)
 stream=torch.cuda.current_stream(x.device)
 ct.launch(stream,(groups,),partial_eq,(flat,partial,n,constant))
 ct.launch(stream,(1,),finish_eq,(partial,out,groups,width))
 return out.item()==0
