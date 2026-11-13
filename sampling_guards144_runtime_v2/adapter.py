from . import range_adapter as ranged
from .owned import prepare
from .checks import Checks
ATTR=ranged.ATTR;ORDER=ranged.ORDER
class Greedy(ranged.Greedy):
 def install(self,model):
  result=super().install(model)
  self.candidate,self.sampling_provenance=prepare(self.candidate,'greedy_api',self.controller.sampling96.predicate,self.controller.sampling96.gather)
  result['sampling_copy']=self.sampling_provenance;return result
 def apply(self,v,ctx):
  out=super().apply(v,ctx)
  # Original temperature check has already succeeded; observe metadata only.
  self.controller.sampling96.observe_temperature(v['temperature'],v['expanded_idx_mapping'])
  return out
class Sampler(ranged.Sampler):
 def install(self,model):
  self.sampling_bindings=[];result=super().install(model);env=self.candidate.__globals__;old=env['_tileir_selected_ranks']
  d=old.__kwdefaults__;assert set(d)=={'_body','_site'}and d['_site']=='ranks_selected'
  body,proof=prepare(d['_body'],'ranks_api',self.controller.sampling96.predicate,self.controller.sampling96.gather)
  assert body.__globals__['_range_guard']is d['_body'].__globals__['_range_guard']and body.__globals__['_finite_all']is d['_body'].__globals__['_finite_all']
  def wrapper(*a,_body=body,_site='ranks_selected',**k):return self.guarded_body(_body,(_site,),*a,**k)
  self.sampling_bindings.append((env,'_tileir_selected_ranks',old,wrapper));env['_tileir_selected_ranks']=wrapper
  result['sampling_copy']=proof;return result
 def close(self):
  errors=[]
  for env,key,old,new in reversed(getattr(self,'sampling_bindings',[])):
   if env[key]is new:env[key]=old
   else:errors.append(key)
  result=super().close()
  if errors:raise RuntimeError('Owned sampling body restore conflict',errors)
  return result
SPECS=tuple((k,{'greedy_sample':Greedy,'sampler_four':Sampler}.get(k,c),n)for k,c,n in ranged.SPECS)
class Controller(ranged.Controller):
 def __init__(self,model):
  ranged.base.Controller.__init__(self,model,SPECS);self.range48=ranged.Checks(self);self.sampling96=Checks(self)
 def start_mode(self,mode):
  self.sampling96.start(mode)
  try:return super().start_mode(mode)
  except BaseException as e:self.latch('sampling96-start',e);raise
 def finish(self):
  sample=self.sampling96.finish();out=super().finish();out['sampling96']=sample;return out
def install(model):
 assert not hasattr(model,ATTR);c=Controller(model);setattr(model,ATTR,c);return c.install()
def start(model,mode):return getattr(model,ATTR).start_mode(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):return ranged.close(model)
def release(model):return ranged.release(model)
