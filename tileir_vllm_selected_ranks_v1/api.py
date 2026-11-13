"""Selected-token competition ranks (>=), not full sampling or topk."""
def reference_rows(rows,ids):
 import math
 assert len(rows)==len(ids)
 out=[]
 for row,i in zip(rows,ids):
  assert type(i)is int and 0<=i<len(row) and math.isfinite(row[i])
  assert all(not math.isnan(v) and v!=float('inf') for v in row)
  out.append(sum(v>=row[i] for v in row))
 return out
def validate_descriptors(x,ids):
 assert x['shape']==[32,248320] and x['stride']==[248320,1] and x['storage_offset']==0 and x['dtype']=='torch.bfloat16'
 assert ids['shape']==[32] and ids['stride']==[1] and ids['storage_offset']==0 and ids['dtype']=='torch.int64'
 assert not x['requires_grad'] and not ids['requires_grad']
def selected_token_ranks(logits,token_ids):
 import torch,cuda.tile as ct
 from .kernels import partial,finish
 assert isinstance(logits,torch.Tensor) and isinstance(token_ids,torch.Tensor)
 assert logits.device.type=='cuda' and token_ids.device==logits.device
 def meta(t):return dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),dtype=str(t.dtype),requires_grad=t.requires_grad)
 validate_descriptors(meta(logits),meta(token_ids))
 assert bool(((token_ids>=0)&(token_ids<248320)).all())
 chosen=logits.gather(1,token_ids[:,None]);assert bool(torch.isfinite(chosen).all())
 assert not bool(torch.isnan(logits).any()) and not bool(torch.isposinf(logits).any())
 c=torch.empty((32,31),device=logits.device,dtype=torch.int32);out=torch.empty((32,),device=logits.device,dtype=torch.int64)
 stream=torch.cuda.current_stream(logits.device);ct.launch(stream,(32,31),partial,(logits,token_ids,c));ct.launch(stream,(32,),finish,(c,out));return out
