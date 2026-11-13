"""Public TorchDispatchMode observer; no tensor values/storage contents read."""
import json,traceback
class Ledger:
 def __init__(self,max_calls=200000,max_sites=10000,max_bytes=32*1024*1024):
  self.max_calls=max_calls;self.max_sites=max_sites;self.max_bytes=max_bytes
  self.calls=0;self.sites={};self.rows=[];self.bytes=0;self.failed=False
 def site(self,op,metadata,frames):
  assert not self.failed,'Observer failure latch'
  key=json.dumps([op,metadata,frames],sort_keys=True)
  if self.calls>=self.max_calls or (key not in self.sites and(len(self.rows)>=self.max_sites or self.bytes+len(key.encode())>self.max_bytes)):
   self.failed=True;raise RuntimeError('Metadata inventory cap before original operation; HOLD')
  self.calls+=1
  if key not in self.sites:
   self.sites[key]=len(self.rows);self.rows.append(dict(op=op,metadata=metadata,frames=frames,calls=0,returns=0,errors=0));self.bytes+=len(key.encode())
  idx=self.sites[key];self.rows[idx]['calls']+=1;return idx
 def snapshot(self):return dict(calls=self.calls,sites=self.rows,failed=self.failed,metadata_bytes=self.bytes,helper_counts=getattr(self,'helper_counts',{}),helpers_restored=getattr(self,'helpers_restored',False),scope='Shapes/layout/storage metadata and Python source frames only; no payload, no timing')
def create_mode(ledger):
 import torch
 from torch.utils._python_dispatch import TorchDispatchMode
 def desc(x,depth=0):
  assert depth<=8,'ATen metadata nesting cap'
  if isinstance(x,torch.Tensor):
   return dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),device=str(x.device),offset=x.storage_offset(),layout=str(x.layout),requires_grad=x.requires_grad)
  if isinstance(x,(list,tuple)):
   if len(x)>128:return dict(python_type=type(x).__name__,length=len(x),contents_omitted=True)
   return [desc(v,depth+1)for v in x]
  if isinstance(x,dict):
   assert len(x)<=128 and all(type(k)is str for k in x)
   return {k:desc(v,depth+1)for k,v in x.items()}
  # Scalar argument values deliberately not dumped: avoids payload-derived scalar capture.
  return dict(python_type=type(x).__name__)
 class Observer(TorchDispatchMode):
  def __torch_dispatch__(self,func,types,args=(),kwargs=None):
   kw=kwargs or {};frames=[dict(file=f.filename,line=f.lineno,function=f.name)for f in traceback.extract_stack(limit=24)[:-1] if '/public-tileir-blas-'in f.filename or '/vllm/'in f.filename][-12:]
   idx=ledger.site(str(func),dict(args=desc(args),kwargs=desc(kw)),frames)
   try:
    with torch.profiler.record_function('metadata_site::'+str(idx)):result=func(*args,**kw)
   except BaseException:
    ledger.failed=True;ledger.rows[idx]['errors']+=1;raise
   ledger.rows[idx]['returns']+=1
   return result # exact original returned object; do not inspect values or clone.
 return Observer()
