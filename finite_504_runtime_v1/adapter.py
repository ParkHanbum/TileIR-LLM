"""504 predicate shadow/replacement on an otherwise unchanged lean26 finite472."""
from collections import Counter
from lean26_gdn_finite_runtime_v1 import adapter as base
from lean_public_composition_v1.recurrent import PostConv as OriginalPost
from lean_public_composition_v1.metadata import KV as OriginalKV
from lean_public_composition_v2.attention_runner import AttentionCore as OriginalAttention
from floating_guard_504_design_v1.factory import copied_method
ATTR=base.ATTR
ORDER=base.ORDER
LIMITS={**{'post.'+k:24 for k in ('conv_output','a','b','A_log','dt_bias')},'kv.key':128,'kv.value':128,'attention.out':128}
class Checks:
 def __init__(self,controller):self.controller=controller;self.active=False;self.mode=None;self.rows=[];self.counts=Counter();self.shadow_pass=False;self.busy=False
 def start(self,mode):
  assert not self.active and self.controller.failed is None and mode in ('shadow','replace')
  assert mode!='replace'or self.shadow_pass
  self.mode=mode;self.rows=[];self.counts=Counter();self.active=True
 def finite(self,site,x):
  import torch
  from tileir_validation_primitives_v3.api import all_finite
  from tileir_kv_strided_finite_v1.api import all_finite_value
  try:
   assert self.active and self.controller.active and self.controller.failed is None and not self.busy
   assert site in LIMITS and self.counts[site]<LIMITS[site]
   self.busy=True
   meta=lambda:(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.data_ptr(),x.is_neg(),x.is_conj())
   desc=meta();storage=x.untyped_storage();assert storage.nbytes()<=128*1024*1024
   def snapshot():return torch.empty(0,dtype=torch.uint8,device=x.device).set_(storage,0,(storage.nbytes(),),(1,)).cpu()
   before=snapshot();stream=torch.cuda.current_stream(x.device).cuda_stream
   native=bool(torch.isfinite(x).all())if self.mode=='shadow'else None
   value=all_finite_value(x)if site=='kv.value'else all_finite(x)
   assert native is None or native is value,'Finite boolean mismatch'
   assert meta()==desc and torch.equal(before,snapshot()),'Finite predicate mutated input owner'
   assert torch.cuda.current_stream(x.device).cuda_stream==stream
   self.counts[site]+=1;self.rows.append(dict(site=site,index=self.counts[site],mode=self.mode,native=native,tile=value,input_owner_preserved=True,metadata_preserved=True,same_stream=True,descriptor=desc[:4]))
   # Preserve original assert/AND semantics, including false postcheck after producer.
   return native if self.mode=='shadow'else value
  except BaseException as e:self.controller.latch('finite504',e);raise
  finally:self.busy=False
 def finish(self):
  assert self.active and self.controller.failed is None and dict(self.counts)==LIMITS and len(self.rows)==504
  self.active=False
  if self.mode=='shadow':self.shadow_pass=True
  return dict(mode=self.mode,counts=dict(self.counts),calls=504,dense=376,strided=128,rows=self.rows,native_predicates=504 if self.mode=='shadow'else 0,tile_predicates=504,shadow_pass=self.shadow_pass)
class Post(OriginalPost):
 def install(self,model):return copied_method('post',OriginalPost.install,self.controller.finite504.finite)(self,model)
class KV(OriginalKV):
 def apply(self,ctx,v):return copied_method('kv',OriginalKV.apply,self.controller.finite504.finite)(self,ctx,v)
class Attention(OriginalAttention):
 def apply(self,v,ctx):return copied_method('attention',OriginalAttention.apply,self.controller.finite504.finite)(self,v,ctx)
SPECS=tuple((k,{'post_conv':Post,'attention_kv':KV,'attention_core':Attention}.get(k,c),n)for k,c,n in base.SPECS)
class Controller(base.Controller):
 def __init__(self,model):super().__init__(model,SPECS);self.finite504=Checks(self)
 def start_mode(self,mode):
  self.finite504.start(mode)
  try:return super().start()
  except BaseException as e:self.latch('start',e);raise
 def finish(self):
  checks=self.finite504.finish();r=super().finish();r['finite504']=checks;return r

def install(model):
 assert not hasattr(model,ATTR);c=Controller(model);setattr(model,ATTR,c)
 try:return c.install()
 except BaseException:
  # Original installer already attempts reverse cleanup. Retain failed owner.
  raise

def start(model,mode):return getattr(model,ATTR).start_mode(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):return base.close(model)
def release(model):return base.release(model)
