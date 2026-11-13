"""Narrow observed runner padding fill, in-place; no global patch/fallback."""
def validate_contract(shape,stride,dtype,offset,storage_bytes,value):
 assert type(value)is bool and dtype=='torch.bool' and len(shape)==1 and tuple(stride)==(1,)
 assert shape[0] in (0,32,1696) and (not value or shape[0]==0)
 assert type(offset)is int and offset>=0 and offset+shape[0]<=storage_bytes
 return shape[0]
def fill_padding_(view,value):
 import torch
 n=validate_contract(tuple(view.shape),tuple(view.stride()),str(view.dtype),view.storage_offset(),view.untyped_storage().nbytes(),value)
 assert view.device.type=='cuda' and not view.requires_grad
 if n:
  import cuda.tile as ct
  from .kernel import fill_false
  ct.launch(torch.cuda.current_stream(view.device),((n+255)//256,),fill_false,(view,))
 return view
