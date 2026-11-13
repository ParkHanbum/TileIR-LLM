"""Functional public score-helper copy replaces ranks only, not Triton launch ABI."""
from pathlib import Path
import inspect,json,hashlib
from block_gather_runtime_capture_v1.observer import phase_metadata
from tp1_embedding_runtime_capture_v2.observer import meta
from norm_runtime_contract_capture_v1.observer import identity,aliases
from block_gather_runtime_integration_v1.adapter import bytecopy
ATTR='_selected_rank_runtime_v1';ROOT=Path(__file__).resolve().parent
def census(rows):return len(rows)==16 and [x['phase'] for x in rows]==['prefill']+['decode']*15
def build_copy(module,rank_fn):
 spec=json.loads((ROOT/'SOURCE.json').read_text());assert hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()==spec['sha256']
 source=(ROOT/'copied_scores.py.txt').read_bytes();assert hashlib.sha256(source).hexdigest()==spec['copy_sha256']
 env=dict(module.__dict__,_tileir_selected_ranks=rank_fn);exec(compile(source,str(ROOT/'copied_scores.py.txt'),'exec'),env);return env['compute_topk_scores']
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.bindings=[];self.pending=None;self.rows=[];self.mode='shadow';self.shadow_pass=False;self.native_gate_calls=0;self.tileir_gate_calls=0;self.first_mismatch=None
 def install_classes(self,S,module):
  original_call=S.__call__;fn=module.compute_topk_scores
  def call(owner,logits,batch,*args,**kwargs):
   assert self.pending is None
   ctx=dict(batch=phase_metadata(batch),calls=0);self.pending=ctx
   try:
    assert owner.logprobs_mode=='raw_logprobs' and owner.trace_replay_state is None
    ret=original_call(owner,logits,batch,*args,**kwargs);assert ctx['calls']==1;return ret
   finally:self.pending=None
  def helper(*args,**kwargs):
   ctx=self.pending;assert ctx is not None;ctx['calls']+=1;assert ctx['calls']==1
   bound=inspect.signature(fn).bind(*args,**kwargs);bound.apply_defaults()
   return self.apply(bound.arguments,lambda:fn(*args,**kwargs),ctx['batch'])
  for obj,name,old,new in [(S,'__call__',original_call,call),(module,'compute_topk_scores',fn,helper)]:
   self.bindings.append((obj,name,old,new));setattr(obj,name,new)
 def install(self,model):
  from vllm.v1.worker.gpu.sample import sampler,logprob
  from tileir_vllm_selected_ranks_v1 import selected_token_ranks
  assert sampler.compute_topk_scores is logprob.compute_topk_scores
  source=identity(sampler.compute_topk_scores);spec=json.loads((ROOT/'SOURCE.json').read_text());assert source['source_sha256']==spec['sha256']
  self.copy=build_copy(logprob,selected_token_ranks);self.install_classes(sampler.Sampler,sampler)
  return dict(source=source,copy_sha256=spec['copy_sha256'],scope='Rank-only functional copy; native topk/selected-logprob prefix retained')
 def apply(self,v,native,batch):
  import torch
  from tileir_vllm_selected_ranks_v1 import selected_token_ranks
  from tileir_vllm_selected_ranks_v1.api import validate_descriptors
  assert self.mode in ('shadow','replace') and (self.mode!='replace' or self.shadow_pass)
  assert len(self.rows)<16 and batch['num_reqs']==32 and batch['phase']==('prefill' if len(self.rows)==0 else 'decode')
  assert v['num_logprobs']==5 and v['max_per_req_token_ids']==0 and v['logits_mode'] is False and v['cu_num_logits'] is None
  x,ids=v['logits'],v['sampled_token_ids'];validate_descriptors(meta(x),meta(ids))
  tensors={k:t for k,t in v.items() if isinstance(t,torch.Tensor)};before={k:bytecopy(t) for k,t in tensors.items()}
  if self.mode=='shadow':ret=native();self.native_gate_calls+=1;candidate=selected_token_ranks(x,ids)
  else:ret=self.copy(**v);candidate=ret.selected_token_ranks
  self.tileir_gate_calls+=1;torch.cuda.synchronize();rank=ret.selected_token_ranks
  raw=True if self.mode=='replace' else torch.equal(rank.view(torch.uint8),candidate.view(torch.uint8))
  layout=all(t.dtype==torch.int64 and tuple(t.shape)==(32,) and t.stride()==(1,) and t.storage_offset()==0 for t in (rank,candidate))
  valid=all(bool(((t>=1)&(t<=248320)).all()) for t in (rank,candidate))
  preserved=all(torch.equal(before[k],bytecopy(t)) for k,t in tensors.items())
  fresh=all(not z['storage_overlap'] for y in (rank,candidate) for z in aliases({**tensors,'rank':y}).values())
  prefix=ret.logprob_token_ids.shape==(32,6) and ret.logprobs.shape==(32,6) and ret.logprobs.dtype==torch.float32 and bool(torch.isfinite(ret.logprobs).all()) and torch.equal(ret.logprob_token_ids[:,0],ids)
  passed=raw and layout and valid and preserved and fresh and bool(prefix)
  row=dict(mode=self.mode,phase=batch['phase'],batch=batch,inputs={k:meta(t) for k,t in tensors.items()},raw_integer_equal=raw if self.mode=='shadow' else None,raw_reference='same-call native rank' if self.mode=='shadow' else None,layout_equal=layout,input_full_storage_preserved=preserved,fresh_disjoint=fresh,output_valid_range=valid,native_topk_logprobs_retained=True,passed=passed)
  self.rows.append(row)
  if not passed:
   bank=self.output/'first_mismatch';bank.mkdir();torch.save(dict(inputs=before,native_or_actual=rank.cpu(),candidate=candidate.cpu()),bank/'values.pt');self.first_mismatch=dict(row=row,path=str(bank/'values.pt'),sha256=hashlib.sha256((bank/'values.pt').read_bytes()).hexdigest())
  self.save();assert passed,'Selected rank contract/raw HOLD';return ret
 def start(self,mode):
  assert mode in ('shadow','replace') and (mode!='replace' or self.shadow_pass)
  self.mode=mode;self.rows=[];self.native_gate_calls=self.tileir_gate_calls=0;return dict(mode=mode)
 def summary(self):return dict(mode=self.mode,rows=self.rows,supported_calls=len(self.rows),native_gate_calls=self.native_gate_calls,tileir_gate_calls=self.tileir_gate_calls,shadow_pass=self.shadow_pass,first_mismatch=self.first_mismatch,supported_vendor_fallback_calls=0,scope='Counts refer to rank suboperation only; native topk/logprob retained')
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
