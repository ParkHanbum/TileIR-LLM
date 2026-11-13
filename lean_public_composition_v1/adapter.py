"""Prospective lean composition. CPU-authored only; not GPU-qualified or timed."""
from .model import Norm,AttentionGate,GatedNorm,QKRope,Silu,Embedding,Gemm
from .metadata import Gather,Slots,Positions,KV
from .recurrent import PrefillConv,DecodeConv,PostConv,GDNDecode
from .sampler import Counts,Greedy,SamplerThree
ATTR='_lean_public_composition_v1'
SPECS=(('norm',Norm,1040),('attention_gate',AttentionGate,128),('block_gather',Gather,16),('slot_mapping',Slots,16),('pos_seq',Positions,16),('gated_norm',GatedNorm,384),('sample_counts',Counts,16),('qk_rope',QKRope,128),('silu_mul',Silu,512),('indexed_prefill_conv',PrefillConv,24),('indexed_decode_conv',DecodeConv,360),('post_conv',PostConv,24),('embedding',Embedding,16),('attention_kv',KV,128),('greedy_sample',Greedy,16),('gdn_decode_core',GDNDecode,360),('all_gemm',Gemm,2448),('sampler_three',SamplerThree,16))
ORDER=tuple(k for k,c,n in SPECS)
class Controller:
 def __init__(self,model,specs=SPECS):
  self.model=model;self.specs=specs;self.parts=[];self.active=False;self.failed=None;self.owner_refs=[];self.sampler_owner_refs=[]
 def latch(self,key,error):
  if self.failed is None:self.failed=dict(component=key,error=repr(error))
 def install(self):
  results={}
  try:
   for key,cls,count in self.specs:
    part=cls(self,key,count);self.parts.append(part);results[key]=part.install(self.model)
   return dict(components=results,install_order=[x.key for x in self.parts],expected_boundaries=5648,expected_logical_operations=5680,qualified=False,scope='Lean binding candidate only; public API checks and supported-scope guards retained')
  except BaseException as e:
   self.latch('install',e)
   try:self.close()
   except BaseException as cleanup:raise RuntimeError('Install failed; cleanup also failed',repr(e),repr(cleanup))from e
   raise
 def start(self):
  assert not self.active and self.failed is None
  for part in self.parts:part.start()
  self.active=True;return dict(mode='lean',validation_adapter_diagnostics=False,public_API_validation=True)
 def finish(self):
  assert self.active and self.failed is None
  try:
   result={x.key:x.finish()for x in self.parts}
   for part in self.parts:
    if not hasattr(part,'owners'):continue
    owners=part.owners
    expected={(name,p):n for name in owners for p,n in ([('prefill',1)]if part.key in('indexed_prefill_conv','post_conv')else[('decode',15)]if part.key in('indexed_decode_conv','gdn_decode_core')else[('prefill',1),('decode',15)])}
    assert dict(part.counts)==expected,(part.key,'owner census')
   for key in('block_gather','slot_mapping','pos_seq'):
    part=next(x for x in self.parts if x.key==key);assert dict(part.counts)=={('runner','prefill'):1,('runner','decode'):15}
   assert sum(x['calls']for x in result.values())==5648
   self.active=False;return dict(components=result,selected_boundaries=5648,selected_logical_operations=5680,selected_native_calls=0,selected_fallback_calls=0,failed=None,scope='Dispatch census only; raw correctness must be checked by a separate qualification harness; not a performance result')
  except BaseException as e:self.latch('finish',e);raise
 def close(self):
  results={};errors=[];order=[]
  for part in reversed(self.parts):
   order.append(part.key)
   try:results[part.key]=part.close()
   except BaseException as e:errors.append((part.key,repr(e)))
  self.parts=[];self.active=False
  # Retain UVA owners even after close until the caller synchronizes/drops this controller.
  assert not errors,errors
  return dict(restored=True,components=results,close_order=order,owner_lifetime='Caller must synchronize device before dropping controller/engine')
def install(model,output=None):
 assert not hasattr(model,ATTR);state=Controller(model);setattr(model,ATTR,state)
 try:return state.install()
 except BaseException:delattr(model,ATTR);raise
def start(model,mode='lean'):
 assert mode=='lean','No shadow bypass or production result inferred'
 return getattr(model,ATTR).start()
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 state=getattr(model,ATTR,None)
 if state is None:return dict(restored=True,already_absent=True)
 # Keep object reachable in model until caller-synchronized release(). No hidden sync here.
 return state.close()
def release(model):
 state=getattr(model,ATTR,None)
 if state is None:return
 assert not state.parts and not state.active,'close and caller device synchronization required before release'
 delattr(model,ATTR)
