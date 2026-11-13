"""24 public prefill core boundaries; native state gather/scatter stays outside."""
from pathlib import Path
import json,inspect,functools
from norm_runtime_contract_capture_v1.observer import metadata,identity
from gdn_post_conv_runtime_capture_v1.observer import phase_meta
ATTR='_tileir_gdn_prefill_core_integration_v3'
NAMES=('q','k','v','g','beta','initial_state','cu_seqlens','chunk_indices','chunk_offsets')
def guarded(fn):
 @functools.wraps(fn)
 def call(self,*args,**kwargs):
  self.guard()
  try:return fn(self,*args,**kwargs)
  except BaseException as error:
   self.latch(fn.__name__,error)
   raise
 return call
class Adapter:
 def __init__(self,output):
  self.out=Path(output);self.out.mkdir(exist_ok=False);self.modules={};self.bindings=[];self.mode='shadow';self.shadow_pass=False;self.baselines={};self.rows=[];self.counts={};self.native_gate_calls=0;self.tileir_gate_calls=0;self.pending=False;self.failed=False;self.first_failure=None;self.failure_context={};self.failure_tensors={}
 def guard(self):
  assert not self.failed,'GDN first failure latched; no further dispatch'
 def snapshot(self,name,t):
  n=t.numel()*t.element_size();total=sum(x.numel()*x.element_size() for x in self.failure_tensors.values())
  assert n<=128*1024*1024 and total+n<=320*1024*1024
  self.failure_tensors[name]=t.detach().cpu().clone()
 def latch(self,boundary,error):
  if self.failed:return
  self.failed=True;self.first_failure=dict(boundary=boundary,error=repr(error),context=self.failure_context.copy())
  bank=self.out/'first_mismatch';bank.mkdir(exist_ok=False)
  try:
   import torch
   total=0;manifest={}
   for name,t in self.failure_tensors.items():
    n=t.numel()*t.element_size();assert n<=128*1024*1024 and total+n<=320*1024*1024
    total+=n;torch.save(t,bank/(name+'.pt'));manifest[name]=dict(shape=list(t.shape),dtype=str(t.dtype),logical_bytes=n)
   self.first_failure['artifacts']=manifest;self.first_failure['logical_bytes']=total
  except BaseException as dump_error:self.first_failure['artifact_error']=repr(dump_error)
  (bank/'FIRST-FAILURE.json').write_text(json.dumps(self.first_failure,indent=2,default=str))
 @guarded
 def install(self,model):
  from vllm.forward_context import get_forward_context
  named=dict(model.named_modules())
  for name,m in named.items():
   if not name.endswith('.linear_attn.chunk_gated_delta_rule'):continue
   owner=named[name.rsplit('.',1)[0]];original=m._forward_method;sig=inspect.signature(original)
   assert str(m.gdn_prefill_backend)=='triton' and original.__name__=='forward_native'
   self.modules[name]=dict(selected=identity(original),backend=str(m.gdn_prefill_backend),owner_prefix=owner.prefix)
   @functools.wraps(original)
   def wrapper(*args,_name=name,_owner=owner,_original=original,_sig=sig,**kwargs):
    self.guard()
    self.failure_context=dict(path=_name,mode=self.mode,boundary='entry');self.failure_tensors={}
    try:
     assert not self.pending;ctx=get_forward_context();assert isinstance(ctx.attn_metadata,dict)
     phase=phase_meta(ctx.attn_metadata.get(_owner.prefix));assert phase['phase']=='prefill' and not phase['speculative'] and phase['num_actual_tokens']==1696
     self.counts[_name]=self.counts.get(_name,0)+1;assert self.counts[_name]==1
     b=_sig.bind(*args,**kwargs);b.apply_defaults();self.pending=True
     try:return self.apply(_name,b.arguments,lambda:_original(*args,**kwargs),phase)
     finally:self.pending=False
    except BaseException as error:
     self.latch('wrapper',error);raise
   had='_forward_method' in m.__dict__;prior=m.__dict__.get('_forward_method');self.bindings.append((m,had,prior,wrapper));m._forward_method=wrapper
  assert len(self.modules)==24;return dict(modules=self.modules,selected=24,scope='prefill core only, state gather/scatter and all other operations native')
 @guarded
 def apply(self,name,v,native,phase):
  import torch
  from gdn_post_conv_qualification_v2.run import compare
  from gdn_prefill_chain_qualification_v1.chain import execute,LABELS
  from tileir_gdn_chunk_cumsum_v1 import chunk_local_cumsum
  from tileir_gdn_chunk_kkt_v5 import chunk_scaled_dot_kkt
  from tileir_gdn_solve_tril_v6 import solve_tril
  from tileir_gdn_wy_v1 import recompute_w_u_fwd
  from tileir_gdn_chunk_state_v1 import chunk_state
  from tileir_gdn_chunk_output_v2 import chunk_output
  from tileir_gdn_chunk_cumsum_v1.contract import validate_metadata
  assert {k:v[k] for k in v if k not in NAMES}==dict(output_final_state=True,use_qk_l2norm_in_kernel=False,core_attn_out=None,aiter_prefill_metadata=None)
  self.failure_context.update(path=name,phase=phase,mode=self.mode,scalar_arguments={k:v[k] for k in v if k not in NAMES})
  inputs={k:v[k].detach() for k in NAMES};descs={k:metadata(t) for k,t in inputs.items()}
  specs={'q':((1,1696,16,128),torch.bfloat16),'k':((1,1696,16,128),torch.bfloat16),'v':((1,1696,32,128),torch.bfloat16),'g':((1,1696,32),torch.float32),'beta':((1,1696,32),torch.float32),'initial_state':((32,32,128,128),torch.float32),'cu_seqlens':((33,),torch.int32),'chunk_indices':((32,2),torch.int32),'chunk_offsets':((33,),torch.int64)}
  for k,t in inputs.items():
   shape,dtype=specs[k];assert t.is_cuda and t.device==inputs['q'].device and tuple(t.shape)==shape and t.dtype==dtype and t.is_contiguous() and t.storage_offset()==0 and t.untyped_storage().nbytes()==t.numel()*t.element_size()<=128*1024*1024
  self.failure_context['input_descriptors']=descs
  for k,t in inputs.items():self.snapshot('input_'+k,t)
  validate_metadata(inputs['cu_seqlens'].cpu().tolist(),inputs['chunk_indices'].cpu().tolist());assert inputs['chunk_offsets'].cpu().tolist()==list(range(33))
  assert all(bool(torch.isfinite(t).all()) for t in inputs.values())
  assert bool((inputs['initial_state']==0).all()),'Nonzero initial state outside qualified fresh-prefill scope; HOLD'
  def digest(t):
   import hashlib
   return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  before={k:digest(t) for k,t in inputs.items()};row=dict(path=name,phase=phase['phase'],mode=self.mode,passed=False,initial_state_all_zero=True,input_descriptors=descs,input_sha256=before,stage_labels=[]);self.rows.append(row);self.progress()
  if self.mode=='replace':assert self.shadow_pass and before==self.baselines[name]['inputs'],'Unqualified live input change after shadow; HOLD'
  if self.mode=='shadow':
   actual=native();self.native_gate_calls+=1;assert isinstance(actual,tuple) and len(actual)==2
   for k,t in zip(('o','final_state'),actual):self.snapshot('native_'+k,t)
   assert all(digest(t)==before[k] for k,t in inputs.items()),'Native input mutation HOLD'
  ops=dict(cumsum=chunk_local_cumsum,kkt=chunk_scaled_dot_kkt,solve=solve_tril,wy=recompute_w_u_fwd,state=chunk_state,output=chunk_output);retained={}
  def overlap(a,b):
   aa=a.untyped_storage();bb=b.untyped_storage();return aa.data_ptr()<bb.data_ptr()+bb.nbytes() and bb.data_ptr()<aa.data_ptr()+aa.nbytes()
  def observe(outputs):
   for label,t in outputs.items():
    assert label not in retained and bool(torch.isfinite(t).all()) and t.is_contiguous() and t.storage_offset()==0 and t.untyped_storage().nbytes()==t.numel()*t.element_size()<=128*1024*1024
    assert all(not overlap(t,z) for z in [*inputs.values(),*retained.values()]);retained[label]=t;row['stage_labels'].append(label)
  with torch.inference_mode():candidate=execute(inputs,ops,observe)
  for k,t in zip(('o','final_state'),candidate):self.snapshot('candidate_'+k,t)
  self.tileir_gate_calls+=1;assert row['stage_labels']==list(LABELS)
  torch.cuda.synchronize();assert candidate[0].dtype==torch.bfloat16 and tuple(candidate[0].shape)==(1,1696,32,128) and candidate[1].dtype==torch.float32 and tuple(candidate[1].shape)==(32,32,128,128)
  out_hash=[digest(t) for t in candidate];preserved=all(digest(t)==before[k] for k,t in inputs.items());row['input_preserved']=preserved;row['candidate_output_sha256']=out_hash
  if self.mode=='shadow':
   metrics={k:compare(a.detach().cpu(),b.cpu(),torch) for k,a,b in zip(('o','final_state'),actual,candidate)};passed=all(c['raw_equal'] and c['strict_failures']==c['reference_nonfinite']==c['candidate_nonfinite']==0 for c in metrics.values());row['metrics']=metrics;self.baselines[name]=dict(inputs=before,outputs=out_hash);ret=actual
  else:
   passed=out_hash==self.baselines[name]['outputs'];row['shadow_output_raw_equal']=passed;ret=candidate
  row['passed']=bool(passed and preserved);self.progress()
  if not row['passed']:
   raise AssertionError('GDN prefill core raw/strict/finite/storage HOLD')
  self.failure_tensors={};self.failure_context={}
  return ret
 @guarded
 def start(self,mode):
  assert mode in ('shadow','replace') and (mode!='replace' or self.shadow_pass);self.mode=mode;self.rows=[];self.counts={};self.native_gate_calls=self.tileir_gate_calls=0;return dict(mode=mode)
 def progress(self):
  r=dict(failed=self.failed,first_failure=self.first_failure,mode=self.mode,rows=self.rows,supported_calls=len(self.rows),native_gate_calls=self.native_gate_calls,tileir_gate_calls=self.tileir_gate_calls,shadow_pass=self.shadow_pass,supported_vendor_fallback_calls=0,scope='24 fresh zero-initial prefill cores only; state gather/scatter/decode/other ops native');(self.out/(self.mode+'-PROGRESS.json')).write_text(json.dumps(r,indent=2));return r
 @guarded
 def finish(self):
  assert not self.pending and self.counts=={n:1 for n in self.modules};assert len(self.rows)==24 and all(x['passed'] for x in self.rows) and self.tileir_gate_calls==24 and self.native_gate_calls==(24 if self.mode=='shadow' else 0)
  if self.mode=='shadow':self.shadow_pass=True
  return self.progress()
 def close(self):
  conflicts=[]
  for m,had,old,new in reversed(self.bindings):
   if m.__dict__.get('_forward_method') is not new:conflicts.append(type(m).__name__);continue
   if had:m._forward_method=old
   else:delattr(m,'_forward_method')
  self.bindings=[];assert not conflicts;return dict(restored=True)
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
