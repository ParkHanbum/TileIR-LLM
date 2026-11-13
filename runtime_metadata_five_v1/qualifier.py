# SPDX-License-Identifier: Apache-2.0
from . import api
LIMITS={"subq":1,"subs":1,"gt":1,"full":2}
class Qualifier:
 def __init__(self,torch,save=lambda:None):self.torch=torch;self.save=save;self.failed=False;self.rows=[];self.mode=None;self.shadow_pass=False;self.current=None;self.binding=None
 def start(self,mode):
  if self.failed or mode not in ("shadow","replace") or (mode=="replace" and not self.shadow_pass):self.failed=True;raise RuntimeError("phase/latch")
  self.mode=mode;self.rows=[];self.current=None
 def raw(self,x):return x.detach().contiguous().view(self.torch.uint8).cpu().numpy().tobytes()
 def backing(self,x):
  s=x.untyped_storage()
  if s.nbytes()>1048576:raise ValueError("backing cap")
  v=self.torch.empty(0,dtype=self.torch.uint8,device=x.device).set_(s,0,(s.nbytes(),),(1,))
  return self.raw(v)
 def guard(self,fn):
  def wrapped(*args,**kwargs):
   if self.failed:raise RuntimeError("latched entry")
   try:return fn(*args,**kwargs)
   except BaseException:self.failed=True;self.save();raise
  return wrapped
 def invoke(self,label,inputs,native,candidate,device):
  if self.failed or self.mode is None or sum(r["label"]==label for r in self.rows)>=LIMITS[label]:self.failed=True;raise RuntimeError("latch/cap")
  self.current={"label":label,"native_attempts":0,"native_returns":0,"candidate_attempts":0,"candidate_returns":0};self.save()
  try:
   initial_stream=self.torch.cuda.current_stream(device).cuda_stream
   before=[self.backing(x) for x in inputs];meta=[(tuple(x.shape),tuple(x.stride()),x.storage_offset(),x.data_ptr(),x.untyped_storage().data_ptr()) for x in inputs]
   self.current["native_attempts"]+=1;self.save();ref=native();self.current["native_returns"]+=1
   self.current["candidate_attempts"]+=1;self.save();out=candidate();self.current["candidate_returns"]+=1
   stream=self.torch.cuda.current_stream(out.device)
   if stream.cuda_stream!=initial_stream:raise ValueError("stream changed")
   stream.synchronize()
   if out.dtype!=ref.dtype or out.shape!=ref.shape or self.raw(out)!=self.raw(ref):raise ValueError("raw mismatch")
   if any(self.backing(x)!=old or (tuple(x.shape),tuple(x.stride()),x.storage_offset(),x.data_ptr(),x.untyped_storage().data_ptr())!=m for x,old,m in zip(inputs,before,meta)):raise ValueError("input mutation")
   if out.untyped_storage().data_ptr()==ref.untyped_storage().data_ptr() or any(out.untyped_storage().data_ptr()==x.untyped_storage().data_ptr() for x in inputs):raise ValueError("output alias")
   self.rows.append(dict(label=label,raw=True,input_preserved=True,output_fresh=True,native_oracle_calls=1,candidate_calls=1,production_native_return=self.mode=="shadow"));self.save();return ref if self.mode=="shadow" else out
  except BaseException:self.failed=True;self.save();raise
 def subq(self,a,b):
  api.validate(a);api.validate(b)
  return self.invoke("subq",(a,b),lambda:a-b,lambda:api.subtract(a,b),a.device)
 def subs(self,a,b):
  api.validate(a);api.validate(b)
  return self.invoke("subs",(a,b),lambda:a-b,lambda:api.subtract(a,b),a.device)
 def gt(self,a):
  api.validate(a,(0,))
  return self.invoke("gt",(a,),lambda:a>0,lambda:api.positive(a),a.device)
 def full(self,size,fill_value,**kwargs):
  api.validate_full(size,fill_value,**kwargs)
  return self.invoke("full",(),lambda:self.torch.full(size,fill_value,**kwargs),lambda:api.full(size,fill_value,**kwargs),kwargs["device"])
 def finish(self):
  if self.failed or {k:sum(r["label"]==k for r in self.rows)for k in LIMITS}!=LIMITS:self.failed=True;raise RuntimeError("census")
  if self.mode=="shadow":self.shadow_pass=True
  return dict(mode=self.mode,calls=5,counts=dict(LIMITS),raw_all=True,native_oracle_calls=5,candidate_calls=5,production_native_returns=5 if self.mode=="shadow" else 0)
 def close(self):
  if self.binding:self.binding.close()
