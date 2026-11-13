# SPDX-License-Identifier: Apache-2.0
# Public metadata implementation. Same qualified numeric API; no native oracle.
from runtime_metadata_five_v1 import api
from runtime_metadata_five_v1.binding import Bindings
LIMITS={"subq":1,"subs":1,"gt":1,"full":2}
def descriptor(x):return (str(x.dtype),str(x.device),tuple(x.shape),tuple(x.stride()),x.storage_offset(),x.data_ptr(),x.untyped_storage().data_ptr(),x.untyped_storage().nbytes(),x.requires_grad,x.is_neg(),x.is_conj())
class Controller:
 def __init__(self,current_stream=None):self.current_stream=current_stream;self.failed=False;self.counts={k:0 for k in LIMITS};self.mode=None;self.binding=None;self.closed=False
 def stream(self,device):
  if self.current_stream:return self.current_stream(device)
  import torch
  return torch.cuda.current_stream(device)
 def guard(self,fn):
  def wrapped(*args,**kwargs):
   if self.failed:raise RuntimeError("latched metadata entry")
   try:return fn(*args,**kwargs)
   except BaseException:self.failed=True;raise
  return wrapped
 def start(self,mode='lean'):
  if self.failed or self.closed or mode!='lean':self.failed=True;raise RuntimeError("phase/latch")
  self.mode=mode;self.counts={k:0 for k in LIMITS}
 def invoke(self,label,inputs,call,device,dtype,shape):
  if self.failed or self.mode!='lean' or self.counts[label]>=LIMITS[label]:self.failed=True;raise RuntimeError("latch/cap")
  try:
   before=[descriptor(x)for x in inputs];stream=self.stream(device).cuda_stream
   out=call()
   if self.stream(device).cuda_stream!=stream:raise ValueError("stream changed")
   if str(out.dtype)!=dtype or out.device!=device or tuple(out.shape)!=shape or tuple(out.stride())!=(1,) or out.storage_offset()!=0:raise ValueError("output contract")
   if out.requires_grad or out.is_neg() or out.is_conj():raise ValueError("output flags")
   if [descriptor(x)for x in inputs]!=before:raise ValueError("input metadata changed")
   if any(out.untyped_storage().data_ptr()==x.untyped_storage().data_ptr()for x in inputs):raise ValueError("output alias")
   self.counts[label]+=1;return out
  except BaseException:self.failed=True;raise
 def subq(self,a,b):
  api.validate(a);api.validate(b)
  return self.invoke('subq',(a,b),lambda:api.subtract(a,b),a.device,'torch.int32',(32,))
 def subs(self,a,b):
  api.validate(a);api.validate(b)
  return self.invoke('subs',(a,b),lambda:api.subtract(a,b),a.device,'torch.int32',(32,))
 def gt(self,a):
  api.validate(a,(0,))
  return self.invoke('gt',(a,),lambda:api.positive(a),a.device,'torch.bool',(32,))
 def full(self,size,value,**kwargs):
  api.validate_full(size,value,**kwargs)
  return self.invoke('full',(),lambda:api.full(size,value,**kwargs),kwargs['device'],'torch.int32',(2048,))
 def finish(self):
  if self.failed or self.counts!=LIMITS:self.failed=True;raise RuntimeError('metadata census')
  return dict(mode='lean',calls=5,counts=dict(self.counts),native_oracle_calls=0,candidate_calls=5,raw_rechecked=False,input_metadata_preserved=True,scope='Qualified numeric APIs; new endpoint proof required')
 def install(self,common,gdn,conv):
  from .gates import qualification
  qualification()
  self.binding=Bindings(common,gdn,conv,self);self.binding.install()
 def close(self):
  if self.binding:self.binding.close()
  self.closed=True
