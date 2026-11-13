from collections import Counter
from lean26_gdn_finite_runtime_v1 import adapter as base
from lean_public_composition_v2.attention_runner import AttentionCore as OriginalAttention
from lean26_gdn_finite_runtime_v1.core import PrefillCore as OriginalGDN
from public_eq_all_guard_v2.owned import prepare
ATTR=base.ATTR;ORDER=base.ORDER
LIMITS={'attention.q':128,'attention.k':128,'attention.v':128,'gdn.initial':24}
class Checks:
 def __init__(self,controller):
  self.controller=controller;self.active=False;self.mode=None;self.rows=[];self.counts=Counter();self.shadow_pass=False;self.busy=False
 def start(self,mode):
  assert not self.active and self.controller.failed is None and mode in ('shadow','replace')
  assert mode!='replace' or self.shadow_pass
  self.mode=mode;self.rows=[];self.counts=Counter();self.active=True
 def predicate(self,site,x,value):
  import torch
  from public_eq_all_guard_v2.api import eq_all
  try:
   assert self.active and self.controller.active and self.controller.failed is None and not self.busy
   assert site in LIMITS and self.counts[site]<LIMITS[site]
   assert value==(0 if site=='gdn.initial' else 1)
   if site=='gdn.initial':assert tuple(x.shape)==(32,32,128,128) and tuple(x.stride())==(524288,16384,128,1) and x.storage_offset()==0
   else:assert tuple(x.shape)==(32,4) and tuple(x.stride())==(0,0)
   assert x.dtype==torch.float32
   self.busy=True
   meta=lambda:(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.data_ptr(),x.is_neg(),x.is_conj())
   desc=meta();storage=x.untyped_storage();assert storage.nbytes()<=128*1024*1024
   def snapshot():return torch.empty(0,dtype=torch.uint8,device=x.device).set_(storage,0,(storage.nbytes(),),(1,)).cpu()
   before=snapshot();stream=torch.cuda.current_stream(x.device).cuda_stream
   native=bool((x==value).all())if self.mode=='shadow'else None
   result=eq_all(x,value)
   assert type(result)is bool and (native is None or native is result),'EQ boolean mismatch'
   assert meta()==desc and torch.equal(before,snapshot()),'EQ input owner changed'
   assert torch.cuda.current_stream(x.device).cuda_stream==stream
   self.counts[site]+=1
   self.rows.append(dict(site=site,index=self.counts[site],mode=self.mode,constant=value,native=native,tile=result,input_owner_preserved=True,metadata_preserved=True,same_stream=True,descriptor=desc[:4],physical_elements=16777216 if site=='gdn.initial' else 1,logical_elements=16777216 if site=='gdn.initial' else 128))
   return native if self.mode=='shadow'else result
  except BaseException as e:self.controller.latch('eq408',e);raise
  finally:self.busy=False
 def finish(self):
  assert self.active and self.controller.failed is None and dict(self.counts)==LIMITS and len(self.rows)==408
  self.active=False
  if self.mode=='shadow':self.shadow_pass=True
  return dict(mode=self.mode,counts=dict(self.counts),calls=408,descale384=384,initial24=24,rows=self.rows,native_predicates=408 if self.mode=='shadow'else 0,tile_predicates=408,shadow_pass=self.shadow_pass)
class Attention(OriginalAttention):
 def apply(self,v,ctx):
  names=iter(('attention.q','attention.k','attention.v'));seen=[]
  def eq(x,c):
   site=next(names);seen.append(site);return self.controller.eq408.predicate(site,x,c)
  try:
   out=prepare(OriginalAttention.apply,'attention',eq)(self,v,ctx)
   assert seen==['attention.q','attention.k','attention.v'];return out
  except BaseException as e:self.controller.latch('eq408',e);raise
class GDN(OriginalGDN):
 def install(self,model):
  result=super().install(model)
  self.eq_original_body=self.body
  self.body=prepare(self.body,'gdn_active',lambda x,c:self.controller.eq408.predicate('gdn.initial',x,c))
  self.eq_body=self.body
  return result
 def close(self):
  error=None
  if hasattr(self,'eq_body'):
   if self.body is not self.eq_body:error=RuntimeError('GDN EQ body binding changed')
   else:self.body=self.eq_original_body
  result=super().close()
  if error is not None:raise error
  return result
SPECS=tuple((k,{'attention_core':Attention,'gdn_prefill_core':GDN}.get(k,c),n)for k,c,n in base.SPECS)
class Controller(base.Controller):
 def __init__(self,model):super().__init__(model,SPECS);self.eq408=Checks(self)
 def start_mode(self,mode):
  self.eq408.start(mode)
  try:return super().start()
  except BaseException as e:self.latch('eq408-start',e);raise
 def finish(self):
  checks=self.eq408.finish();r=super().finish();r['eq408']=checks;return r

def install(model):
 assert not hasattr(model,ATTR);c=Controller(model);setattr(model,ATTR,c);return c.install()
def start(model,mode):return getattr(model,ATTR).start_mode(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):return base.close(model)
def release(model):return base.release(model)
