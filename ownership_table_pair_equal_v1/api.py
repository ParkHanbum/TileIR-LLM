def descriptor(t):return (str(t.dtype),str(t.device),tuple(t.shape),tuple(t.stride()),t.storage_offset(),t.untyped_storage().nbytes(),t.data_ptr(),t.untyped_storage().data_ptr(),t.requires_grad,t.is_neg(),t.is_conj(),str(t.layout))
def validate(a,b):
 if str(a.dtype) not in ('torch.uint64','torch.int64') or a.dtype!=b.dtype or a.device!=b.device or a.device.type!='cuda':raise ValueError('typed integer pair')
 if len(a.shape)!=1 or not 1<=a.shape[0]<=4 or tuple(a.shape)!=tuple(b.shape):raise ValueError('table length')
 for t in (a,b):
  if str(t.layout)!='torch.strided' or tuple(t.stride())!=(1,) or t.storage_offset()!=0 or t.untyped_storage().nbytes()!=a.shape[0]*8 or t.requires_grad or t.is_neg() or t.is_conj():raise ValueError('exact table layout/flags')
 return int(a.shape[0])
def equal(a,b):
 import torch
 if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor):raise TypeError("owned torch.Tensor pair required")
 n=validate(a,b);before=descriptor(a),descriptor(b)
 import cuda.tile as ct
 from .kernel import pair_equal
 stream=torch.cuda.current_stream(a.device);out=torch.empty((1,),dtype=torch.int32,device=a.device)
 ct.launch(stream,(1,),pair_equal,(a,b,out,n))
 status=int(out.item()) # Retain host boolean/error boundary and stream lifetime.
 if not 0<=status<=n or (descriptor(a),descriptor(b))!=before or torch.cuda.current_stream(a.device).cuda_stream!=stream.cuda_stream:raise ValueError('status/metadata/stream')
 return status==0
