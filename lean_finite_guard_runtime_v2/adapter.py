"""Lean25 CPU draft. Not GPU-qualified; original numerical APIs unchanged."""
from lean_public_composition_v1.adapter import Controller as OriginalController,SPECS as ORIGINAL_SPECS
from lean_finite_guard_runtime_v2.bindings import SamplerFour,SamplerFP32,HiddenGather
from lean_public_composition_v2.core_parent import PrefillCore,GDNParent
from lean_public_composition_v2.attention_runner import AttentionCore,RunnerPadding,AcceptedScatter
ATTR='_lean_finite_guard_v2'
SPECS=tuple(('sampler_four',SamplerFour,n)if k=='sampler_three'else(k,c,n)for k,c,n in ORIGINAL_SPECS)+(
 ('gdn_prefill_core',PrefillCore,24),('attention_core',AttentionCore,128),('runner_padding',RunnerPadding,16),('hidden_gather',HiddenGather,16),('sampler_fp32',SamplerFP32,16),('accepted_scatter',AcceptedScatter,16),('gdn_parent_state',GDNParent,24))
ORDER=tuple(k for k,c,n in SPECS)
class Controller(OriginalController):
 def __init__(self,model,specs=SPECS):super().__init__(model,specs)
 def install(self):
  results={}
  try:
   for key,cls,count in self.specs:
    part=cls(self,key,count);self.parts.append(part);results[key]=part.install(self.model)
   return dict(components=results,install_order=[x.key for x in self.parts],expected_boundaries=5888,expected_logical_operations=6056,qualified=False,scope='CPU draft lean25; public numerical/value guards retained; new qualification required')
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
   assert sum(x['calls']for x in result.values())==5888
   assert result['sampler_four']['logical_suboperation_calls']==64
   assert result['runner_padding']['logical_suboperation_calls']==64
   assert result['gdn_parent_state']['logical_suboperation_calls']==96
   self.active=False
   return dict(components=result,selected_boundaries=5888,selected_logical_operations=6056,selected_native_calls=0,selected_fallback_calls=0,failed=None,scope='Dispatch census only, not raw proof/performance/physical GPU coverage')
  except BaseException as e:self.latch('finish',e);raise

def install(model,output=None):
 assert not hasattr(model,ATTR);state=Controller(model);setattr(model,ATTR,state)
 try:return state.install()
 except BaseException:delattr(model,ATTR);raise
def start(model,mode='lean'):
 assert mode=='lean';return getattr(model,ATTR).start()
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 state=getattr(model,ATTR,None)
 if state is None:return dict(restored=True,already_absent=True)
 return state.close()
def release(model):
 state=getattr(model,ATTR,None)
 if state is None:return
 assert not state.parts and not state.active,'close and caller synchronization required'
 delattr(model,ATTR)
