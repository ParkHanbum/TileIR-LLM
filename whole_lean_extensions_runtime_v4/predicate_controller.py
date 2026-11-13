"""Explicit controller extension; unchanged numerical APIs. Endpoint pending."""
from lean26_gdn_finite_runtime_v1.adapter import Controller as Base, SPECS, ATTR
from lean_finite_504_v1.api import Checks as Finite, replace_specs
from lean_eq408_v1.api import Checks as Equal
from lean_eq408_v1.bridge import combined_attention, GDNBodyBinding
from lean_public_composition_v2.attention_runner import AttentionCore as OriginalAttention
class Attention(OriginalAttention):
 def apply(self,v,ctx):
  return combined_attention(OriginalAttention.apply,self.controller.finite504,self.controller.eq408)(self,v,ctx)
class Controller(Base):
 def __init__(self,model):
  specs=tuple((k,Attention if k=='attention_core' else c,n)for k,c,n in replace_specs(SPECS))
  super().__init__(model,specs);self.finite504=Finite(self);self.eq408=Equal(self);self.eq_body=None
 def install(self):
  try:
   result=super().install()
   part=next(p for p in self.parts if p.key=='gdn_prefill_core')
   self.eq_body=GDNBodyBinding(part,self.eq408);self.eq_body.install()
   result['combined_predicate_body']=True;return result
  except BaseException as e:self.latch('predicate/install',e);raise
 def start(self):
  try:self.finite504.start();self.eq408.start();return super().start()
  except BaseException as e:self.latch('predicate/start',e);raise
 def finish(self):
  try:
   f=self.finite504.finish();q=self.eq408.finish();result=super().finish()
   result['finite504']=f;result['eq408']=q;return result
  except BaseException as e:self.latch('predicate/finish',e);raise
 def close(self):
  errors=[]
  if self.eq_body is not None:
   try:self.eq_body.close()
   except BaseException as e:errors.append(e)
  try:result=super().close()
  except BaseException as e:errors.append(e);result={}
  if errors:raise RuntimeError('Predicate restore HOLD') from errors[0]
  result['eq_body_restored']=True;return result
