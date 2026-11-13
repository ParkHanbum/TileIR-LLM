"""Wall endpoint unchanged; profile-only execution range excludes proof/install/restore."""
import time
from functools import partial
from contextlib import nullcontext
from lean_public_runtime_qualification_v1.gates import exact_tokens,exact_public,dispatch
from lean_public_runtime_qualification_v1.cleanup import release_checked
from packed_vllm_runtime_compare_v4.gates import public_logprobs
from .profile_contract import scope_names

def absent(model,adapter):return not hasattr(model,adapter.ATTR)
def request(engine,torch,policy,adapter,lifecycle,ids,params,label,mode,measure,frozen_tokens,frozen_public,record,clock=time.perf_counter_ns,profiled=False):
 assert not(profiled and measure),'Profiler requests must never be wall samples'
 names=scope_names(mode)
 def phase(key):return torch.profiler.record_function(names[key])if profiled else nullcontext()
 with phase('install'):
  assert mode in('native','lean')and not engine.has_unfinished_requests()
  assert engine.apply_model(partial(absent,adapter=adapter))==[True]
  _,precision=policy.select('native')
  item=dict(label=label,mode=mode,measured=measure,selected_dispatch=None,whole_ms=None,allow_bf16_reduced_precision_reduction=precision,allow_bf16_reduced_precision_reduction_split_k=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
  if mode=='lean':
   lifecycle['installed']=True
   item['adapter_install']=engine.apply_model(adapter.install)
   engine.apply_model(partial(adapter.start,mode='lean'))
  keys=[label+':'+str(i)for i in range(32)];outputs={}
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 with phase('execution'):
  begin=clock()if measure else None
  for key in keys:engine.add_request(key,{'prompt_token_ids':list(ids)},params)
  while engine.has_unfinished_requests():
   for o in engine.step():
    if o.finished:
     assert o.request_id in keys and o.request_id not in outputs and len(o.outputs)==1
     outputs[o.request_id]=o
  tokens=[list(outputs[k].outputs[0].token_ids)for k in keys]
  torch.cuda.synchronize()
  end=clock()if measure else None
 # Range entry/exit is outside the existing timer. No instrumentation on wall requests.
 with phase('proof'):
  item.update(whole_ms=None if begin is None else(end-begin)/1e6,output_token_ids=tokens,public_logprobs=public_logprobs(outputs,keys),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  if profiled:item['profile_scope_names']=names
  record(item) # preserve outputs even if a later exact gate fails
  item['frozen_tokens_gate']=exact_tokens(tokens,frozen_tokens);item['frozen_public_gate']=exact_public(item['public_logprobs'],frozen_public)
  if mode=='lean':
   census=engine.apply_model(adapter.finish);assert len(census)==1
   item['selected_dispatch']=census[0];item['dispatch_gate']=dispatch(census[0])
 with phase('restore'):
  if mode=='lean':
   torch.cuda.synchronize();item['adapter_restore']=engine.apply_model(adapter.close)
   torch.cuda.synchronize();item['adapter_release_absent']=engine.apply_model(partial(release_checked,adapter=adapter));assert item['adapter_release_absent']==[True]
   lifecycle['installed']=False
  assert not engine.has_unfinished_requests()
  record(item);return item
