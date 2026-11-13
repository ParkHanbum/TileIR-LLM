from gdn_integer_finite_typed_v1.core import PrefillCore as Typed
from lean26_gdn_finite_runtime_v1.core import PrefillCore as Original

def verify_finite_binding(part):
 method=part.body.__globals__.get('_gdn_finite')
 if getattr(method,'__self__',None) is not part or getattr(method,'__func__',None) is not Typed.finite:raise RuntimeError('active owned body finite binding mismatch')
 return True
class PrefillCore(Typed):
 def install(self,model):
  from .gates import qualification
  qualification();result=super().install(model);verify_finite_binding(self);return result
 def start(self):
  try:verify_finite_binding(self);return super().start()
  except BaseException as e:self.controller.latch(self.key,e);raise
 def finish(self):
  try:verify_finite_binding(self);return super().finish()
  except BaseException as e:self.controller.latch(self.key,e);raise

def replace_specs(specs):
 matches=[(k,c,n) for k,c,n in specs if k=='gdn_prefill_core']
 if len(matches)!=1 or matches[0][1] is not Original or matches[0][2]!=24:raise ValueError('exact original finite GDN specification required')
 return tuple((k,PrefillCore if k=='gdn_prefill_core' else c,n)for k,c,n in specs)
