"""Six local finite predicates; numeric bodies and original global bindings immutable."""
from types import MethodType
from finite_six_site_cpu_v1.factory import build
from finite_six_site_cpu_v1.predicate import finite_boolean
from lean_finite_guard_runtime_v1.sampler import SamplerFour as Top5Qualified
from lean_public_composition_v2.sampler import SamplerFP32 as CopyOriginal
from lean_public_composition_v2.attention_runner import HiddenGather as HiddenOriginal
SITES=('hidden_post','sampler_pre','copy_api','sampler_post','scores_api','ranks_selected')
class FiniteSites:
 def setup_finite(self):
  from tileir_validation_primitives_v2.api import require_finite
  self.finite_scope=None;self.finite_impl_six=require_finite;self.finite_counts={s:[0,0]for s in self.site_names}
 def start(self):
  super().start();assert self.finite_scope is None;self.finite_counts={s:[0,0]for s in self.site_names}
 def finite_all(self,site,x):
  try:
   assert self.controller.active and self.controller.failed is None,'Finite guard outside active request or after failure'
   assert self.finite_scope is not None,'Finite guard outside owned body'
   allowed,seen=self.finite_scope
   assert site in allowed and site not in seen,'Unexpected or duplicate finite site in owned body'
   assert site in self.finite_counts and self.finite_counts[site][0]<16,'Finite site census exceeded'
   seen.add(site);self.finite_counts[site][0]+=1
   ok=finite_boolean(x,self.finite_impl_six)
   if ok:self.finite_counts[site][1]+=1
   return ok
  except BaseException as e:self.controller.latch(self.key,e);raise
 def guarded_body(self,fn,allowed,*a,**k):
  try:
   self.guard() # external entry retains the strict helper census bound
   assert self.finite_scope is None,'Nested finite owned body'
   self.finite_scope=(set(allowed),set())
   out=fn(*a,**k)
   assert self.finite_scope[1]==set(allowed),'Missing finite site'
   return out
  except BaseException as e:self.controller.latch(self.key,e);raise
  finally:self.finite_scope=None
 def finish(self):
  assert self.finite_scope is None
  out=super().finish()
  assert set(self.finite_counts)==set(self.site_names)
  assert all(v==[16,16]for v in self.finite_counts.values())
  out['finite_sites']={s:dict(attempts=v[0],passes=v[1],original_selected_predicate_calls=0,host_status_item_retained=True)for s,v in self.finite_counts.items()}
  return out
class HiddenGather(FiniteSites,HiddenOriginal):
 site_names=('hidden_post',)
 def install(self,model):
  self.setup_finite();body=build('hidden_post',HiddenOriginal.gather,self.finite_all)
  self.gather=lambda *a,**k:self.guarded_body(body,self.site_names,self,*a,**k)
  return super().install(model)
class SamplerFP32(FiniteSites,CopyOriginal):
 site_names=('sampler_pre','copy_api','sampler_post')
 def install(self,model):
  self.setup_finite();body=build('sampler_outer',CopyOriginal.copy_logits,self.finite_all)
  self.copy_logits=lambda *a,**k:self.guarded_body(body,self.site_names,self,*a,**k)
  result=super().install(model)
  self.api=build('copy_api',self.api,self.finite_all)
  return result
class SamplerFour(FiniteSites,Top5Qualified):
 site_names=('scores_api','ranks_selected')
 def install(self,model):
  self.setup_finite();result=super().install(model)
  env=self.candidate.__globals__
  for key,name in (('_tileir_selected_logprobs','scores_api'),('_tileir_selected_ranks','ranks_selected')):
   old=env[key];body=build(name,old,self.finite_all)
   def wrapper(*a,_body=body,_site=name,**k):return self.guarded_body(_body,(_site,),*a,**k)
   env[key]=wrapper;assert env[key]is not old
  return result
