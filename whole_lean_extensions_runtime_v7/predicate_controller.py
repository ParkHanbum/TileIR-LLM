"""Explicit controller extension; unchanged numerical APIs. Endpoint pending."""
from lean26_gdn_finite_runtime_v1.adapter import Controller as Base, SPECS, ATTR
from lean_finite_504_v1.api import Checks as Finite, replace_specs
from lean_eq408_v1.api import Checks as Equal
from lean_temperature16_v1.adapter import Checks as Temperature, Greedy as TemperatureGreedy, annotate_sampling_receipt
from lean_sampling144_v1.adapter import Range, Sampling, proof as sampling_proof
from gdn_integer_finite_lean_v1.core import replace_specs as integer_specs
from gdn_integer_finite_runtime_qualification_v3.adapter import Controller as LifetimeController
from lean_eq408_v1.bridge import combined_attention, GDNBodyBinding
from lean_public_composition_v2.attention_runner import AttentionCore as OriginalAttention
class Attention(OriginalAttention):
 def apply(self,v,ctx):
  return combined_attention(OriginalAttention.apply,self.controller.finite504,self.controller.eq408)(self,v,ctx)
class Controller(Base):
 def latch(self,key,error):
  from .failure import record
  if getattr(self,"primary_failure",None)is None:self.primary_failure=record(key,error)
  return super().latch(key,error)
 def __init__(self,model):
  specs=tuple((k,Attention if k=='attention_core' else c,n)for k,c,n in integer_specs(replace_specs(SPECS)))
  assert sum(k=='greedy_sample' for k,c,n in specs)==sum(k=='sampler_four' for k,c,n in specs)==1
  specs=tuple((k,{'greedy_sample':TemperatureGreedy,'sampler_four':sampling_proof.Sampler}.get(k,c),n)for k,c,n in specs)
  super().__init__(model,specs);self.finite504=Finite(self);self.eq408=Equal(self);self.eq_body=None
  self.range48=Range(self);self.sampling96=Sampling(self);self.temperature16=Temperature(self)
 def install(self):
  try:
   result=super().install()
   part=next(p for p in self.parts if p.key=='gdn_prefill_core')
   self.eq_body=GDNBodyBinding(part,self.eq408);self.eq_body.install()
   result['combined_predicate_body']=True;return result
  except BaseException as e:self.latch('predicate/install',e);raise
 def start(self):
  try:self.finite504.start();self.eq408.start();self.range48.start('replace');self.sampling96.start('replace');self.temperature16.start();return super().start()
  except BaseException as e:self.latch('predicate/start',e);raise
 def finish(self):
  try:
   f=self.finite504.finish();q=self.eq408.finish();r=self.range48.finish();s=self.sampling96.finish();t=self.temperature16.finish();s=annotate_sampling_receipt(s,t);result=super().finish()
   result['finite504']=f;result['eq408']=q;result['range48']=r;result['sampling96']=s;result['temperature16']=t;return result
  except BaseException as e:self.latch('predicate/finish',e);raise
 def close(self):
  if getattr(self,"_permanent_close_hold",False):raise RuntimeError("Permanent restoration HOLD")
  self._lifetime_parts=list(self.parts)
  errors=[]
  if self.eq_body is not None:
   try:self.eq_body.close()
   except BaseException as e:errors.append(e)
  try:result=LifetimeController.close(self)
  except BaseException as e:errors.append(e);result={}
  if errors:
   self._permanent_close_hold=True;self._restoration_proven=False
   raise RuntimeError('Predicate restore HOLD') from errors[0]
  result['eq_body_restored']=True;return result
