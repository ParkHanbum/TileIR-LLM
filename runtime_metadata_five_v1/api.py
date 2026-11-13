# SPDX-License-Identifier: Apache-2.0
# Scope: actual int32 rank-one metadata; views may alias each other, never outputs.
def validate(x,offsets=(0,1)):
 if str(x.dtype)!="torch.int32" or x.device.type!="cuda" or tuple(x.shape)!=(32,) or tuple(x.stride())!=(1,) or x.storage_offset() not in offsets or x.requires_grad or x.is_neg() or x.is_conj():raise ValueError("unsupported metadata tensor")
 if x.untyped_storage().nbytes()>1048576 or (x.storage_offset()+32)*4>x.untyped_storage().nbytes():raise ValueError("owner extent")
def subtract(a,b):
 validate(a);validate(b)
 if a.device!=b.device:raise ValueError("device")
 import torch,cuda.tile as ct
 from .kernel import sub32
 out=torch.empty((32,),dtype=torch.int32,device=a.device)
 ct.launch(torch.cuda.current_stream(a.device),(1,),sub32,(a,b,out))
 return out
def positive(a):
 validate(a,(0,))
 import torch,cuda.tile as ct
 from .kernel import gt32
 out=torch.empty((32,),dtype=torch.bool,device=a.device)
 ct.launch(torch.cuda.current_stream(a.device),(1,),gt32,(a,out))
 return out
def validate_full(size,fill_value,*,dtype,device):
 if tuple(size)!=(2048,) or type(fill_value)is not int or fill_value!=-1 or str(dtype)!="torch.int32" or getattr(device,"type",None)!="cuda":raise ValueError("unsupported full metadata")
def full(size,fill_value,*,dtype,device):
 validate_full(size,fill_value,dtype=dtype,device=device)
 import torch,cuda.tile as ct
 from .kernel import full2048
 out=torch.empty((2048,),dtype=dtype,device=device)
 ct.launch(torch.cuda.current_stream(device),(8,),full2048,(out,))
 return out
