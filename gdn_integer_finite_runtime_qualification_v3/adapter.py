"""Lean25 CPU draft. Not GPU-qualified; original numerical APIs unchanged."""
from lean_public_composition_v1.adapter import Controller as OriginalController,SPECS as ORIGINAL_SPECS
from lean_finite_guard_runtime_v4.bindings import SamplerFour,SamplerFP32
from gdn_integer_finite_runtime_qualification_v3.core import PrefillCore
from lean_public_composition_v2.core_parent import GDNParent
from lean_public_composition_v2.attention_runner import AttentionCore,RunnerPadding,AcceptedScatter
from lean_finite_guard_runtime_v4.bindings import HiddenGather
ATTR='_integer_finite72_qualification_v3'
SPECS=tuple(('sampler_four',SamplerFour,n)if k=='sampler_three'else(k,c,n)for k,c,n in ORIGINAL_SPECS)+(
 ('gdn_prefill_core',PrefillCore,24),('attention_core',AttentionCore,128),('runner_padding',RunnerPadding,16),('hidden_gather',HiddenGather,16),('sampler_fp32',SamplerFP32,16),('accepted_scatter',AcceptedScatter,16),('gdn_parent_state',GDNParent,24))
from lean26_prefill_composition_v1.prefill import PrefillInputs,check_owned_bridge
SPECS=(('prepare_prefill_inputs',PrefillInputs,1),)+SPECS
ORDER=tuple(k for k,c,n in SPECS)
class Controller(OriginalController):
 def __init__(self,model,specs=SPECS):super().__init__(model,specs)
 def close(self):
  # A failed production Base.close can erase its records. Never retry/promote it.
  if getattr(self,'_permanent_close_hold',False):raise RuntimeError('permanent restoration HOLD; owners retained')
  self._binding_evidence=[(part,tuple(getattr(part,'bindings',()))) for part in self.parts]
  # Keep every owner strongly reachable, including partially installed failures.
  if not hasattr(self,'_lifetime_parts'):self._lifetime_parts=list(self.parts)
  results={};errors=[];order=[];failed=[]
  self._restoration_proven=False
  for part in reversed(self.parts):
   order.append(part.key)
   try:
    receipt=part.close()
    if not isinstance(receipt,dict) or receipt.get('restored') is not True:raise RuntimeError('close lacks restored receipt')
    results[part.key]=receipt
   except BaseException as e:errors.append((part.key,repr(e)));failed.append(part)
  self.parts=list(reversed(failed));self.active=False
  if errors:
   self._permanent_close_hold=True
   raise RuntimeError('permanent partial close HOLD',errors)
  self._restoration_proven=True
  return dict(restored=True,components=results,close_order=order,owner_lifetime='Caller must synchronize device before dropping controller/engine')
 def install(self):
  results={}
  try:
   for key,cls,count in self.specs:
    part=cls(self,key,count);self.parts.append(part);results[key]=part.install(self.model)
   prefill=self.parts[0];padding=next(p for p in self.parts if p.key=='runner_padding');assert check_owned_bridge(padding,prefill.wrapped)
   return dict(components=results,owned_prefill_bridge_verified=True,install_order=[x.key for x in self.parts],expected_boundaries=5889,expected_logical_operations=6057,qualified=False,scope='CPU draft lean26; public numerical/value guards retained; new qualification required')
  except BaseException as e:
   self.latch('install',e)
   try:self.close()
   except BaseException as cleanup:raise RuntimeError('Install and cleanup failed',repr(e),repr(cleanup))from e
   raise
 def finish(self):
  assert self.active and self.failed is None
  try:
   result={x.key:x.finish()for x in self.parts}
   for part in self.parts:
    if part.key in ('gdn_prefill_core','attention_core','runner_padding','hidden_gather','sampler_fp32','accepted_scatter','gdn_parent_state'):continue # Explicit owner/phase gates in each new part.
    if not hasattr(part,'owners'):continue
    expected={(name,p):n for name in part.owners for p,n in ([('prefill',1)]if part.key in('indexed_prefill_conv','post_conv')else[('decode',15)]if part.key in('indexed_decode_conv','gdn_decode_core')else[('prefill',1),('decode',15)])}
    assert dict(part.counts)==expected,(part.key,'owner census')
   for key in ('block_gather','slot_mapping','pos_seq'):
    part=next(x for x in self.parts if x.key==key);assert dict(part.counts)=={('runner','prefill'):1,('runner','decode'):15}
   assert sum(x['calls']for x in result.values())==5889
   assert result['sampler_four']['logical_suboperation_calls']==64
   assert result['runner_padding']['logical_suboperation_calls']==64
   assert result['gdn_parent_state']['logical_suboperation_calls']==96
   self.active=False
   return dict(components=result,selected_boundaries=5889,selected_logical_operations=6057,selected_native_calls=0,selected_fallback_calls=0,failed=None,scope='Dispatch census only, not raw proof/performance/physical GPU coverage')
  except BaseException as e:self.latch('finish',e);raise

def install(model,output=None):
 assert not hasattr(model,ATTR);state=Controller(model);setattr(model,ATTR,state)
 try:return state.install()
 except BaseException:
  # Deliberately retain ATTR: outer finally must retry/record restoration.
  raise
def start(model,mode='lean'):
 assert mode in ('shadow','lean');state=getattr(model,ATTR);state.integer_phase=mode;return state.start()
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 state=getattr(model,ATTR,None)
 if state is None:return dict(restored=True,already_absent=True)
 return state.close()
def release(model):
 state=getattr(model,ATTR,None)
 if state is None:return
 assert not state.parts and not state.active and getattr(state,'_restoration_proven',False) and not getattr(state,'_permanent_close_hold',False),'proven close and caller synchronization required'
 delattr(model,ATTR)
