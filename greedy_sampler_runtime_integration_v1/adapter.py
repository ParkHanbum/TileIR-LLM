"""Only public gumbel_sample greedy boundary, top5/logprob pipeline untouched."""
from pathlib import Path
import inspect,json,hashlib
from block_gather_runtime_capture_v1.observer import phase_metadata
from tp1_embedding_runtime_capture_v2.observer import meta
from norm_runtime_contract_capture_v1.observer import identity,aliases
from block_gather_runtime_integration_v1.adapter import bytecopy
ATTR='_greedy_sampler_integration_v1'
def census(rows):return len(rows)==16 and [x['phase'] for x in rows]==['prefill']+['decode']*15
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.bindings=[];self.pending=None;self.rows=[];self.mode='shadow';self.shadow_pass=False;self.native_gate_calls=0;self.tileir_gate_calls=0;self.first_mismatch=None
 def install_classes(self,S,module):
  original_call=S.__call__;fn=module.gumbel_sample
  def call(owner,logits,batch,*args,**kwargs):
   assert self.pending is None
   ctx=dict(batch=phase_metadata(batch),calls=0);self.pending=ctx
   try:
    assert owner.logprobs_mode=='raw_logprobs' and owner.trace_replay_state is None
    ret=original_call(owner,logits,batch,*args,**kwargs);assert ctx['calls']==1;return ret
   finally:self.pending=None
  def helper(*args,**kwargs):
   ctx=self.pending;assert ctx is not None,'Greedy helper outside sampler context unsupported';ctx['calls']+=1;assert ctx['calls']==1
   bound=inspect.signature(fn).bind(*args,**kwargs);bound.apply_defaults()
   return self.apply(bound.arguments,lambda:fn(*args,**kwargs),ctx['batch'])
  for obj,name,old,new in [(S,'__call__',original_call,call),(module,'gumbel_sample',fn,helper)]:
   self.bindings.append((obj,name,old,new));setattr(obj,name,new)
 def install(self,model):
  from vllm.v1.worker.gpu.sample import sampler
  source=identity(sampler.gumbel_sample)
  assert source['source_sha256']=='c14a5fa1899a9ff320c62f29bd161be4a4f0fa124ecf26a28681f0f3d143868e'
  self.install_classes(sampler.Sampler,sampler);return dict(source=source,scope='Only temperature-zero gumbel boundary; top5 unchanged')
 def apply(self,values,native,batch):
  import torch
  from tileir_vllm_greedy_sample_v1 import greedy_sample
  from tileir_vllm_greedy_sample_v1.api import validate_descriptor
  assert self.mode in ('shadow','replace') and (self.mode!='replace' or self.shadow_pass)
  assert batch['num_reqs']==32 and batch['phase']==('prefill' if len(self.rows)==0 else 'decode') and len(self.rows)<16
  assert values['apply_temperature'] is False and values['is_drafting'] is False and values['use_fp64'] is False
  assert all(values[k] is None for k in ('logits_cache','logits_cache_col','logits_cache_source'))
  tensors={k:values[k] for k in ('logits','expanded_idx_mapping','temperature','seed','pos')};x=tensors['logits'];validate_descriptor(meta(x))
  idx=tensors['expanded_idx_mapping'];assert idx.dtype==torch.int64 and tuple(idx.shape)==(32,) and bool(((idx>=0)&(idx<len(tensors['temperature']))).all())
  assert bool((tensors['temperature'].index_select(0,idx)==0).all())
  before={k:bytecopy(t) for k,t in tensors.items()}
  if self.mode=='shadow':out=native();self.native_gate_calls+=1;candidate=greedy_sample(x)
  else:out=greedy_sample(x);candidate=out
  self.tileir_gate_calls+=1;torch.cuda.synchronize()
  raw=True if self.mode=='replace' else torch.equal(out.view(torch.uint8),candidate.view(torch.uint8))
  layout=all(t.dtype==torch.int64 and tuple(t.shape)==(32,) and t.stride()==(1,) and t.storage_offset()==0 for t in (out,candidate))
  valid=all(bool(((t>=0)&(t<248320)).all()) for t in (out,candidate))
  preserved=all(torch.equal(before[k],bytecopy(t)) for k,t in tensors.items())
  fresh=all(not z['storage_overlap'] for y in (out,candidate) for z in aliases({**tensors,'output':y}).values())
  passed=raw and layout and valid and preserved and fresh
  row=dict(mode=self.mode,phase=batch['phase'],batch=batch,inputs={k:meta(t) for k,t in tensors.items()},outputs=[meta(out),meta(candidate)],raw_integer_equal=raw if self.mode=='shadow' else None,raw_reference='same-call native' if self.mode=='shadow' else None,output_valid_integer_range=valid,layout_equal=layout,input_full_storage_preserved=preserved,fresh_disjoint=fresh,passed=passed)
  self.rows.append(row)
  if not passed:
   bank=self.output/'first_mismatch';bank.mkdir();torch.save(dict(inputs=before,native_or_actual=out.cpu(),candidate=candidate.cpu()),bank/'values.pt');self.first_mismatch=dict(row=row,path=str(bank/'values.pt'),sha256=hashlib.sha256((bank/'values.pt').read_bytes()).hexdigest())
  self.save();assert passed,'Greedy sampled-token contract/raw HOLD';return out
 def start(self,mode):
  assert mode in ('shadow','replace') and (mode!='replace' or self.shadow_pass)
  self.mode=mode;self.rows=[];self.native_gate_calls=self.tileir_gate_calls=0;return dict(mode=mode)
 def summary(self):return dict(mode=self.mode,rows=self.rows,supported_calls=len(self.rows),native_gate_calls=self.native_gate_calls,tileir_gate_calls=self.tileir_gate_calls,shadow_pass=self.shadow_pass,first_mismatch=self.first_mismatch,supported_vendor_fallback_calls=0)
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2)+'\n')
 def finish(self):
  assert census(self.rows) and all(x['passed'] for x in self.rows) and self.tileir_gate_calls==16 and self.native_gate_calls==(16 if self.mode=='shadow' else 0)
  if self.mode=='shadow':self.shadow_pass=True
  self.save();return self.summary()
 def close(self):
  errors=[]
  for obj,name,old,new in reversed(self.bindings):
   try:assert getattr(obj,name) is new;setattr(obj,name,old)
   except BaseException as e:errors.append(repr(e))
  self.bindings=[];assert not errors,errors;return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR);a=Adapter(output)
 try:r=a.install(model);setattr(model,ATTR,a);return r
 except BaseException:a.close();raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 r=a.close();delattr(model,ATTR);return r
