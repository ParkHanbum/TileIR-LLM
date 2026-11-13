# SPDX-License-Identifier: Apache-2.0
N=224;SOURCE_BYTES=1792;DESTINATION_BYTES=8192

def check(dst,src,non_blocking=True):
 if non_blocking is not True:raise ValueError('non_blocking scope')
 for x in (src,dst):
  if tuple(x.shape)!=(N,) or tuple(x.stride())!=(1,) or x.storage_offset()!=0 or x.requires_grad or x.is_neg() or x.is_conj():raise ValueError('tensor layout/flags')
 if str(src.dtype)!='torch.int64' or src.device.type!='cpu' or src.is_pinned() is not True:raise ValueError('actual pinned int64 source required')
 if str(dst.dtype)!='torch.int32' or dst.device.type!='cuda':raise ValueError('destination type/device')
 if src.untyped_storage().nbytes()!=SOURCE_BYTES or dst.untyped_storage().nbytes()!=DESTINATION_BYTES:raise ValueError('actual backing extent')
 if src.data_ptr()%8 or dst.data_ptr()%4:raise ValueError('alignment')
 if src.untyped_storage().data_ptr()==dst.untyped_storage().data_ptr():raise ValueError('alias')

def descriptor(x):return (str(x.dtype),str(x.device),tuple(x.shape),tuple(x.stride()),x.storage_offset(),x.data_ptr(),x.untyped_storage().data_ptr(),x.untyped_storage().nbytes(),x.requires_grad,x.is_neg(),x.is_conj())
def low32(v):return (v+2**31)%2**32-2**31
