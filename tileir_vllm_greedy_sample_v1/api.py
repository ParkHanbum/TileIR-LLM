"""Inference-only finite-or-negative-infinity greedy selection; no RNG/topk API."""
def validate_descriptor(m):
 assert m['shape']==[32,248320] and m['stride']==[248320,1] and m['storage_offset']==0
 assert m['dtype']=='torch.float32' and not m['requires_grad']
def reference_rows(rows):
 import math
 out=[]
 for row in rows:
  assert row and not any(math.isnan(v) or v==float('inf') for v in row)
  assert any(math.isfinite(v) for v in row)
  out.append(max(range(len(row)),key=lambda i:row[i]))
 return out
def greedy_sample(logits):
 import torch,cuda.tile as ct
 from .kernels import blocks,finish
 assert isinstance(logits,torch.Tensor) and logits.device.type=='cuda'
 validate_descriptor(dict(shape=list(logits.shape),stride=list(logits.stride()),storage_offset=logits.storage_offset(),dtype=str(logits.dtype),requires_grad=logits.requires_grad))
 # Masked -Inf is valid. NaN/+Inf and all-masked rows are outside this first contract.
 assert not bool(torch.isnan(logits).any()) and not bool(torch.isposinf(logits).any())
 assert bool(torch.isfinite(logits).any(dim=-1).all())
 v=torch.empty((32,243),dtype=torch.float32,device=logits.device)
 i=torch.empty((32,243),dtype=torch.int32,device=logits.device)
 out=torch.empty((32,),dtype=torch.int64,device=logits.device)
 stream=torch.cuda.current_stream(logits.device)
 ct.launch(stream,(32,243),blocks,(logits,v,i));ct.launch(stream,(32,),finish,(v,i,out))
 return out
