"""Finite BF16 fixed-shape top5 ID selection; not torch.topk ABI."""
def validate_descriptor(m):
 assert m['shape']==[32,248320] and m['stride']==[248320,1] and m['storage_offset']==0
 assert m['dtype']=='torch.bfloat16' and not m['requires_grad']
def reference_rows(rows):
 from .reference import reference
 return [reference(row) for row in rows]
def top5_indices(logits):
 import torch,cuda.tile as ct
 from .kernels import blocks,finish
 assert isinstance(logits,torch.Tensor) and logits.device.type=='cuda'
 validate_descriptor(dict(shape=list(logits.shape),stride=list(logits.stride()),storage_offset=logits.storage_offset(),dtype=str(logits.dtype),requires_grad=logits.requires_grad))
 assert bool(torch.isfinite(logits).all()),'NaN/Inf outside finite-only contract'
 values=torch.empty((32,1215),dtype=torch.int32,device=logits.device)
 indices=torch.empty((32,1215),dtype=torch.int32,device=logits.device)
 out=torch.empty((32,5),dtype=torch.int64,device=logits.device)
 stream=torch.cuda.current_stream(logits.device)
 ct.launch(stream,(32,243),blocks,(logits,values,indices));ct.launch(stream,(32,),finish,(values,indices,out))
 return out
