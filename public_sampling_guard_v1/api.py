"""Readonly guards. Each returns at the original host predicate boundary."""
from .contract import matrix,masks,MODES
def _matrix(x):
 import torch
 if not isinstance(x,torch.Tensor)or x.device.type!='cuda'or x.layout!=torch.strided or x.requires_grad:raise ValueError('Readonly CUDA matrix required')
 return matrix(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.untyped_storage().nbytes(),x.is_neg(),x.is_conj())
def classify(x,mode):
 import torch
 if mode not in MODES:raise ValueError('Unknown predicate')
 rows,cols,size=_matrix(x)
 if rows==0:return mode=='rows_have_finite'
 if cols==0:return False
 # Bit views preserve every NaN payload; metadata-only aliases, never contiguous copies.
 raw=x.view(torch.uint32 if size==4 else torch.uint16)
 import cuda.tile as ct
 from .kernel import classify_partial,classify_finish
 groups=(cols+1023)//1024
 p=torch.empty((rows,groups),dtype=torch.int32,device=x.device)
 o=torch.empty((1,),dtype=torch.int32,device=x.device)
 stream=torch.cuda.current_stream(x.device);absolute,inf=masks(str(x.dtype))
 ct.launch(stream,(rows,groups),classify_partial,(raw,p,cols,MODES[mode],absolute,inf))
 ct.launch(stream,(1,),classify_finish,(p,o,rows,groups,1<<(rows-1).bit_length(),1<<(groups-1).bit_length(),MODES[mode]))
 count=int(o.item())
 limit=rows if mode=='rows_have_finite'else rows*cols
 if not 0<=count<=limit:raise RuntimeError('Invalid classification status')
 return count==0 if mode=='rows_have_finite'else count!=0
def has_nan(x):return classify(x,'has_nan')
def has_posinf(x):return classify(x,'has_posinf')
def rows_have_finite(x):return classify(x,'rows_have_finite')
def selected_columns_checked(x,ids):
 """Typed exact gather. Bounds checked before any dependent read.
 Duplicate caller bounds checking intentionally retained in this first standalone API.
 """
 import torch
 rows,cols,size=_matrix(x)
 if not isinstance(ids,torch.Tensor)or ids.device!=x.device or ids.dtype!=torch.int64 or tuple(ids.shape)!=(rows,)or ids.requires_grad:raise ValueError('Matching readonly int64 IDs required')
 from public_i64_range_guard_v1.api import all_in_range_i64
 if not all_in_range_i64(ids,0,cols):raise ValueError('Selected column out of bounds')
 out=torch.empty((rows,1),dtype=x.dtype,device=x.device)
 if rows:
  import cuda.tile as ct
  from .kernel import gather_columns
  bits=torch.uint32 if size==4 else torch.uint16
  ct.launch(torch.cuda.current_stream(x.device),(1,),gather_columns,(x.view(bits),ids,out.view(bits),rows,1<<(rows-1).bit_length()))
 return out
