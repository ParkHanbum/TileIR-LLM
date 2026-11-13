"""Typed, same-itemsize payload copy; reuse existing public copy_bits kernel."""
SHAPE=(1696,32,128)
STRIDE=(4096,128,1)
BYTES=13893632
def metadata(x):
 return dict(shape=list(x.shape),stride=list(x.stride()),offset=x.storage_offset(),dtype=str(x.dtype),device=str(x.device),ptr=x.data_ptr(),storage_ptr=x.untyped_storage().data_ptr(),storage_bytes=x.untyped_storage().nbytes())
def check(dst,src,t):
 if not isinstance(dst,t.Tensor) or not isinstance(src,t.Tensor):raise TypeError('Expected actual Tensor operands')
 for x in (dst,src):
  if tuple(x.shape)!=SHAPE or tuple(x.stride())!=STRIDE or x.dtype!=t.bfloat16 or x.device.type!='cuda' or x.requires_grad or x.is_conj() or x.is_neg():raise ValueError('Unqualified GDN prefill output layout')
  if x.untyped_storage().nbytes()%2 or x.storage_offset()<0 or (x.storage_offset()*2+BYTES)>x.untyped_storage().nbytes() or x.untyped_storage().nbytes()>32*1024*1024:raise ValueError('Unqualified output backing extent')
 if dst.device!=src.device:raise ValueError('Cross-device copy')
 a,b=dst.untyped_storage(),src.untyped_storage()
 if not(a.data_ptr()+a.nbytes()<=b.data_ptr()or b.data_ptr()+b.nbytes()<=a.data_ptr()):raise ValueError('Copy owner overlap')
class Copy:
 def __init__(self):self.failed=False;self.pending=[];self.calls=0
 def __call__(self,dst,src):
  if self.failed:raise RuntimeError('Copy failure latched')
  try:
   import torch
   import cuda.tile as ct
   from tileir_gdn_owned_copies_v1.kernels import copy_bits
   check(dst,src,torch);stream=torch.cuda.current_stream(dst.device)
   s=src.view(torch.uint16).view(-1,32);d=dst.view(torch.uint16).view(-1,32)
   self.pending.append((dst,src,s,d,stream))
   ct.launch(stream,(BYTES//2//1024,1,1),copy_bits,(s,d));self.calls+=1
   return dst
  except BaseException:self.failed=True;raise
 def release_after_sync(self):
  # Caller must finish stream synchronization before relinquishing owners.
  self.pending.clear()
