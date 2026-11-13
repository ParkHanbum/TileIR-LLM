"""Public Sampler helper substitution only after all16 live native shadows pass."""
from pathlib import Path
import json,hashlib
from block_gather_runtime_capture_v1.observer import phase_metadata,metadata
from block_gather_runtime_integration_v1.adapter import bytecopy,clone_storage,same_layout
ATTR='_tileir_sample_counts_integration_v1'
def disjoint(a,b):
 x,y=a.untyped_storage(),b.untyped_storage();return not(x.data_ptr()<y.data_ptr()+y.nbytes() and y.data_ptr()<x.data_ptr()+x.nbytes())
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.bindings=[];self.pending=None;self.mode='shadow';self.rows=[];self.shadow_pass=False;self.native_gate_calls=0;self.tileir_gate_calls=0;self.first_mismatch=None
 def install_classes(self,S,module):
  call=S.__call__;fn=module.get_num_sampled_and_rejected
  def sampler(this,logits,input_batch,*args,**kwargs):
   assert self.pending is None
   ctx=dict(owner=this,batch=input_batch,metadata=phase_metadata(input_batch),calls=0);self.pending=ctx
   try:
    ret=call(this,logits,input_batch,*args,**kwargs);assert ctx['calls']==1;return ret
   finally:self.pending=None
  def counts(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len):
   ctx=self.pending;assert ctx is not None,'Unsupported helper outside sampler boundary';ctx['calls']+=1;b=ctx['batch']
   assert seq_lens is b.seq_lens and cu_num_logits is b.cu_num_logits and idx_mapping is b.idx_mapping
   args=(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len)
   return self.apply(ctx['owner'],args,lambda:fn(*args),ctx['metadata'])
  for obj,name,old,new in ((S,'__call__',call,sampler),(module,'get_num_sampled_and_rejected',fn,counts)):
   self.bindings.append((obj,name,old,new));setattr(obj,name,new)
 def apply(self,sampler,args,native,batch):
  import torch
  from tileir_vllm_sample_counts_v1.api import get_num_sampled_and_rejected,reference_cpu
  assert self.mode in ('shadow','replace') and (self.mode!='replace' or self.shadow_pass)
  sampled,seq,prefix,idx,prefill=args;assert batch['num_reqs']==32 and batch['phase'] in ('prefill','decode') and sampled.shape==(32,)
  owner=sampler.req_states.prefill_len;active=owner.pool._uva_bufs[owner.pool._curr]
  assert prefill is owner.gpu and type(active).__name__=='UvaBuffer' and active.cpu.is_pinned() and prefill.data_ptr()==active.cpu.data_ptr()==active._uva.data_ptr()
  readonly=(*args[1:],owner.cpu,active.cpu);before=[bytecopy(t) for t in readonly];sampled_before=bytecopy(sampled)
  expected=reference_cpu(*(t.detach().cpu().clone() for t in args))
  if self.mode=='shadow':
   cs=clone_storage(sampled);ret=native();self.native_gate_calls+=1
   candidate=get_num_sampled_and_rejected(cs,*args[1:]);self.tileir_gate_calls+=1
  else:
   cs=sampled;ret=get_num_sampled_and_rejected(*args);candidate=ret;self.tileir_gate_calls+=1
  torch.cuda.synchronize(sampled.device)
  assert isinstance(ret,tuple) and isinstance(candidate,tuple) and len(ret)==len(candidate)==2
  actual=[bytecopy(t) for t in ret];cand=[bytecopy(t) for t in candidate]
  exact=[torch.equal(a,b) for a,b in zip(actual,cand)];cpu_exact=[torch.equal(t.detach().cpu(),e) for t,e in zip(ret,expected)];preserved=all(torch.equal(bytecopy(t),b) for t,b in zip(readonly,before))
  identity=ret[0] is sampled and candidate[0] is cs
  allocations=all(disjoint(ret[1],t) for t in (*args,owner.cpu,active.cpu)) and all(disjoint(candidate[1],t) for t in (cs,*args[1:],owner.cpu,active.cpu))
  layouts=all(same_layout(a,b) for a,b in zip(ret,candidate)) and same_layout(ret[0],sampled) and tuple(ret[1].shape)==tuple(sampled.shape) and ret[1].stride()==sampled.stride() and ret[1].dtype==sampled.dtype
  passed=all(exact) and all(cpu_exact) and preserved and identity and allocations and layouts
  row=dict(mode=self.mode,phase=batch['phase'],batch=batch,input_descriptors=[metadata(t) for t in args],prefill_placement=dict(buffer_class=type(active).__name__,host=metadata(active.cpu),view=metadata(prefill)),full_storage_raw_equal=exact,cpu_reference_exact=cpu_exact,readonly_input_preserved=preserved,first_return_same_object=identity,rejected_new_disjoint=allocations,output_layout_equal=layouts,passed=passed)
  self.rows.append(row)
  if not passed and self.first_mismatch is None:
   bank=self.output/'first_mismatch';bank.mkdir();art={}
   for name,value in [('readonly_before',before),('sampled_before',sampled_before),('native_or_actual_after',actual),('candidate_after',cand)]:
    f=bank/(name+'.pt');torch.save(value,f);art[name]=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
   self.first_mismatch=dict(call_index=len(self.rows)-1,row=row,artifacts=art)
  self.save();assert passed,'Sample counts raw/alias/UVA contract HOLD';return ret
 def start(self,mode):
  assert mode in ('shadow','replace') and (mode!='replace' or self.shadow_pass)
  self.mode=mode;self.rows=[];self.native_gate_calls=self.tileir_gate_calls=0;return dict(mode=mode)
 def summary(self):return dict(mode=self.mode,rows=self.rows,supported_calls=len(self.rows),native_gate_calls=self.native_gate_calls,tileir_gate_calls=self.tileir_gate_calls,shadow_pass=self.shadow_pass,first_mismatch=self.first_mismatch,supported_vendor_fallback_calls=0)
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2))
 def finish(self):
  assert len(self.rows)==16 and sum(x['phase']=='prefill' for x in self.rows)==1 and sum(x['phase']=='decode' for x in self.rows)==15 and all(x['passed'] for x in self.rows)
  assert self.native_gate_calls==(16 if self.mode=='shadow' else 0) and self.tileir_gate_calls==16
  if self.mode=='shadow':self.shadow_pass=True
  self.save();return self.summary()
 def close(self):
  conflicts=[]
  for obj,name,old,new in reversed(self.bindings):
   if getattr(obj,name) is new:setattr(obj,name,old)
   else:conflicts.append(name)
  self.bindings=[];assert not conflicts,conflicts;return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR)
 import vllm.v1.worker.gpu.sample.sampler as module
 a=Adapter(output)
 try:a.install_classes(module.Sampler,module);setattr(model,ATTR,a);return dict(boundary='Sampler.__call__/bound get_num_sampled_and_rejected',scope='N32 actual UVA prefill')
 except BaseException:a.close();raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 r=a.close();delattr(model,ATTR);return r
