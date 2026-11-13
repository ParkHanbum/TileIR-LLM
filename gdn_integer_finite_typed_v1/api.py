"""Integer finiteness is constant True; retain a current-stream error boundary."""
def descriptor(t):
 return (str(t.dtype),str(t.device),tuple(t.shape),tuple(t.stride()),t.storage_offset(),t.numel(),t.untyped_storage().nbytes(),t.data_ptr(),t.untyped_storage().data_ptr(),t.requires_grad,t.is_neg(),t.is_conj(),str(t.layout))
def validate(t):
 if str(t.dtype) not in ('torch.int32','torch.int64') or t.device.type!='cuda':raise ValueError('CUDA int32/int64 only')
 if str(t.layout)!='torch.strided' or not t.is_contiguous() or t.storage_offset()!=0 or t.requires_grad or t.is_neg() or t.is_conj():raise ValueError('layout/flags')
 size=4 if str(t.dtype)=='torch.int32' else 8
 if t.numel()<0 or t.untyped_storage().nbytes()!=t.numel()*size or t.untyped_storage().nbytes()>128*1024*1024:raise ValueError('exact backing/cap')
 return descriptor(t)
def all_finite_integer(t,current_stream=None):
 before=validate(t)
 if current_stream is None:
  import torch
  current_stream=torch.cuda.current_stream
 stream=current_stream(t.device);identity=stream.cuda_stream
 stream.synchronize() # Native runtime synchronization remains; launches zero kernels.
 if current_stream(t.device).cuda_stream!=identity or descriptor(t)!=before:raise ValueError('stream/metadata changed')
 return True # Includes empty tensors: all(empty)=True. No integer NaN/Inf encodings.
