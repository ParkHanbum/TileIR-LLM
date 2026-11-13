"""Selected log-softmax scores only; topk ID selection/ranks are separate."""
def validate_descriptors(x,ids):
 assert x['shape']==[32,248320] and x['stride']==[248320,1] and x['storage_offset']==0 and x['dtype']=='torch.bfloat16'
 assert ids['shape']==[32,6] and ids['stride']==[6,1] and ids['storage_offset']==0 and ids['dtype']=='torch.int64'
 assert not x['requires_grad'] and not ids['requires_grad']
def reference_rows(rows,ids):
 import math
 out=[]
 for row,ii in zip(rows,ids):
  assert all(math.isfinite(v) for v in row) and all(type(i)is int and 0<=i<len(row) for i in ii)
  mx=max(row);ls=math.log(sum(math.exp(x-mx) for x in row));out.append([row[i]-mx-ls for i in ii])
 return out
def compute_token_logprobs(logits,token_ids):
 import torch,cuda.tile as ct
 from .kernels import kernel
 assert isinstance(logits,torch.Tensor) and isinstance(token_ids,torch.Tensor) and logits.device.type=='cuda' and token_ids.device==logits.device
 def meta(t):return dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),dtype=str(t.dtype),requires_grad=t.requires_grad)
 validate_descriptors(meta(logits),meta(token_ids));assert bool(torch.isfinite(logits).all()) and bool(((token_ids>=0)&(token_ids<248320)).all())
 out=torch.empty((32,6),dtype=torch.float32,device=logits.device);ct.launch(torch.cuda.current_stream(logits.device),(32,),kernel,(logits,token_ids,out));return out
