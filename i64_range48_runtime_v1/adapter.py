from collections import Counter
from lean26_gdn_finite_runtime_v1 import adapter as base
from lean_finite_guard_runtime_v4.bindings import SamplerFour as OriginalSampler
from lean_public_composition_v1.sampler import Greedy as OriginalGreedy
from .owned import prepare
ATTR=base.ATTR;ORDER=base.ORDER;LIMITS={'greedy':16,'ranks':16,'scores':16}
class Checks:
 def __init__(self,controller):self.controller=controller;self.active=False;self.busy=False;self.shadow_pass=False;self.rows=[]
 def start(self,mode):
  assert not self.active and self.controller.failed is None and mode in ('shadow','replace')and(mode!='replace'or self.shadow_pass)
  self.mode=mode;self.rows=[];self.counts=Counter();self.active=True
 def predicate(self,site,x,lower,upper):
  import torch
  from public_i64_range_guard_v1.api import all_in_range_i64
  from public_i64_range_guard_v1.contract import descriptor
  row=None
  try:
   assert self.active and self.controller.active and self.controller.failed is None and not self.busy
   assert site in LIMITS and self.counts[site]<16 and lower==0
   assert tuple(x.shape)==((32,6)if site=='scores'else(32,))and x.dtype==torch.int64
   assert type(upper)is int and (upper>0 if site=='greedy'else upper==248320)
   descriptor(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.untyped_storage().nbytes(),lower,upper,x.is_neg(),x.is_conj())
   assert x.device.type=='cuda'and not x.requires_grad and x.untyped_storage().nbytes()<=1048576
   self.busy=True;row=dict(site=site,index=self.counts[site]+1,mode=self.mode,lower=lower,upper=upper,stage='pre_snapshot');self.rows.append(row)
   meta=lambda:(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.data_ptr(),x.is_neg(),x.is_conj())
   before_meta=meta();storage=x.untyped_storage()
   def snapshot():return torch.empty(0,dtype=torch.uint8,device=x.device).set_(storage,0,(storage.nbytes(),),(1,)).cpu()
   before=snapshot();stream=torch.cuda.current_stream(x.device).cuda_stream
   row['stage']='native_attempt';native=bool(((x>=lower)&(x<upper)).all())if self.mode=='shadow'else None
   assert torch.equal(before,snapshot()),'Native guard modified input'
   row['stage']='candidate_attempt';value=all_in_range_i64(x,lower,upper);row['stage']='candidate_return'
   assert type(value)is bool and(native is None or value is native),'Range boolean mismatch'
   assert meta()==before_meta and torch.equal(before,snapshot()),'Range input owner changed'
   assert torch.cuda.current_stream(x.device).cuda_stream==stream
   row.update(stage='verified',native=native,tile=value,input_owner_preserved=True,metadata_preserved=True,same_stream=True,descriptor=before_meta[:4])
   self.counts[site]+=1;return native if self.mode=='shadow'else value
  except BaseException as e:
   if row is not None:row['error']=repr(e)
   self.controller.latch('range48',e);raise
  finally:self.busy=False
 def finish(self):
  assert self.active and self.controller.failed is None and dict(self.counts)==LIMITS and len(self.rows)==48 and all(r['stage']=='verified'for r in self.rows)
  self.active=False
  if self.mode=='shadow':self.shadow_pass=True
  return dict(mode=self.mode,calls=48,counts=dict(self.counts),rows=self.rows,native_predicates=48 if self.mode=='shadow'else 0,tile_predicates=48,shadow_pass=self.shadow_pass)
class Greedy(OriginalGreedy):
 def install(self,model):
  self.range_body,self.range_provenance=prepare(OriginalGreedy.apply,'greedy',self.controller.range48.predicate)
  result=super().install(model);result['range_copy']=self.range_provenance;return result
 def apply(self,v,ctx):return self.range_body(self,v,ctx)
class Sampler(OriginalSampler):
 def install(self,model):
  self.range_bindings=[];result=super().install(model);env=self.candidate.__globals__;provenance=[]
  for key,site,finite_site in (('_tileir_selected_ranks','ranks','ranks_selected'),('_tileir_selected_logprobs','scores','scores_api')):
   old=env[key];defaults=old.__kwdefaults__
   assert set(defaults)=={'_body','_site'}and defaults['_site']==finite_site
   body,proof=prepare(defaults['_body'],site,self.controller.range48.predicate);provenance.append(proof)
   assert body.__globals__['_finite_all']is defaults['_body'].__globals__['_finite_all']
   def wrapper(*a,_body=body,_site=finite_site,**k):return self.guarded_body(_body,(_site,),*a,**k)
   self.range_bindings.append((env,key,old,wrapper));env[key]=wrapper
  result['range_copies']=provenance;return result
 def close(self):
  errors=[]
  for env,key,old,new in reversed(getattr(self,'range_bindings',[])):
   if env[key]is new:env[key]=old
   else:errors.append(key)
  result=super().close()
  if errors:raise RuntimeError('Owned range body restore conflict',errors)
  return result
SPECS=tuple((k,{'greedy_sample':Greedy,'sampler_four':Sampler}.get(k,c),n)for k,c,n in base.SPECS)
class Controller(base.Controller):
 def __init__(self,model):super().__init__(model,SPECS);self.range48=Checks(self)
 def start_mode(self,mode):
  self.range48.start(mode)
  try:return super().start()
  except BaseException as e:self.latch('range48-start',e);raise
 def finish(self):
  checks=self.range48.finish();r=super().finish();r['range48']=checks;return r

def install(model):
 assert not hasattr(model,ATTR);c=Controller(model);setattr(model,ATTR,c);return c.install()
def start(model,mode):return getattr(model,ATTR).start_mode(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):return base.close(model)
def release(model):return base.release(model)
