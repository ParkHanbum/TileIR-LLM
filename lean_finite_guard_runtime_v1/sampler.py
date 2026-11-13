"""One existing top5 guard replaced; all other lean25 numerical APIs unchanged."""
from pathlib import Path
import hashlib
from lean_public_composition_v2.sampler import SamplerFour as Original
R=Path(__file__).resolve().parent
class SamplerFour(Original):
 def install(self,model):
  result=super().install(model)
  from tileir_vllm_top5_v2 import top5_indices
  from tileir_validation_primitives_v2.api import require_finite
  assert self.candidate.__globals__['_tileir_top5'] is top5_indices
  self.finite_impl=require_finite;ns={'_finite_guard':self.finite_guard}
  exec(compile((R/'guarded_top5.py.txt').read_text(),str(R/'guarded_top5.py.txt'),'exec'),ns)
  self.guarded_impl=ns['top5_indices'];self.candidate.__globals__['_tileir_top5']=self.top5
  result.update(finite_guard_replaced=True,guarded_source_sha256=hashlib.sha256((R/'guarded_top5.py.txt').read_bytes()).hexdigest(),scope='Only owned top5 finite guard changed; scalar host status remains; not entire native0')
  return result
 def start(self):
  super().start();self.guard_attempts=0;self.guard_passes=0;self.top5_attempts=0;self.top5_returns=0
 def finite_guard(self,x):
  self.guard();self.guard_attempts+=1
  try:
   result=self.finite_impl(x);assert result is None;self.guard_passes+=1;return None
  except BaseException as e:self.controller.latch(self.key,e);raise
 def top5(self,x):
  self.guard();self.top5_attempts+=1
  try:
   before=self.guard_passes;out=self.guarded_impl(x);assert self.guard_passes==before+1;self.top5_returns+=1;return out
  except BaseException as e:self.controller.latch(self.key,e);raise
 def finish(self):
  result=super().finish();assert self.guard_attempts==self.guard_passes==self.top5_attempts==self.top5_returns==16
  result['finite_guard']=dict(attempts=16,passes=16,top5_api_returns=16,original_torch_finite_calls=0,host_scalar_status_retained=True,scope='API counters plus bound source, not physical kernel event counts')
  return result
