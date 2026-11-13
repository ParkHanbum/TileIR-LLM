"""Public class-boundary substitution; no model/source modifications."""
from pathlib import Path
import json,hashlib
from block_gather_runtime_capture_v1.observer import phase_metadata,metadata
ATTR='_tileir_block_gather_integration_v1'
def raw(t):
 import torch
 return torch.empty(0,dtype=torch.uint8,device=t.device).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,))
def bytecopy(t):return raw(t).cpu().clone()
def clone_storage(t):
 b=raw(t).clone();return b.view(t.dtype).as_strided(t.shape,t.stride(),t.storage_offset())
def same_layout(a,b):return a.shape==b.shape and a.stride()==b.stride() and a.storage_offset()==b.storage_offset() and a.dtype==b.dtype
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.bindings=[];self.pending=None;self.mode='shadow';self.rows=[];self.shadow_pass=False;self.native_gate_calls=0;self.tileir_gate_calls=0;self.first_mismatch=None
 def install_classes(self,R,B):
  rp=R.prepare_attn;bg=B.gather_block_tables
  def prepare(r,b,*args,**kwargs):
   assert self.pending is None and r.pcp_manager is None
   ctx=dict(owner=r.block_tables,batch=b,metadata=phase_metadata(b),gathers=0);self.pending=ctx
   try:
    result=rp(r,b,*args,**kwargs);assert ctx['gathers']==1;return result
   finally:self.pending=None
  def gather(owner,idx,num_reqs_padded,out=None,out_ptrs=None):
   ctx=self.pending
   assert ctx is not None and owner is ctx['owner'],'Unexpected gather outside selected boundary'
   ctx['gathers']+=1
   assert out is None and out_ptrs is None and idx is ctx['batch'].idx_mapping and num_reqs_padded==ctx['metadata']['num_reqs_after_padding']
   return self.apply(owner,idx,num_reqs_padded,lambda:bg(owner,idx,num_reqs_padded,out=out,out_ptrs=out_ptrs),ctx['metadata'])
  for cls,name,old,new in ((R,'prepare_attn',rp,prepare),(B,'gather_block_tables',bg,gather)):
   self.bindings.append((cls,name,old,new));setattr(cls,name,new)
 def apply(self,owner,idx,padded,native,batch):
  import torch
  from tileir_vllm_block_gather_v1.api import gather_block_tables,reference_cpu
  assert self.mode in ('shadow','replace') and (self.mode!='replace' or self.shadow_pass)
  assert batch['num_reqs']==padded==32 and batch['phase'] in ('prefill','decode'),'Unqualified actual padding/phase'
  src=tuple(t.gpu for t in owner.block_tables);outputs=tuple(owner.input_block_tables);counts=owner.num_blocks.gpu
  assert [tuple(t.shape) for t in src]==[(32,1)]*3+[(32,8)]
  assert idx.dtype==torch.int64 and tuple(counts.shape)==(4,32)
  pool=owner.num_blocks.pool;active=pool._uva_bufs[pool._curr]
  assert type(active).__name__=='UvaBuffer' and active.cpu.is_pinned() and counts.data_ptr()==active.cpu.data_ptr()==active._uva.data_ptr()
  # Keep active owner refs throughout asynchronous launch and host verification.
  inputs=(*src,counts,idx,active.cpu);before_inputs=[bytecopy(t) for t in inputs];before_outputs=[bytecopy(t) for t in outputs]
  csrc=tuple(t.detach().cpu().clone() for t in src);cc=counts.detach().cpu().clone();ci=idx.detach().cpu().clone();expected=tuple(t.detach().cpu().clone() for t in outputs)
  reference_cpu(csrc,cc,ci,expected,padded)
  if self.mode=='shadow':
   candidates=tuple(clone_storage(t) for t in outputs)
   ret=native();self.native_gate_calls+=1
   candret=gather_block_tables(src,counts,idx,candidates,padded);self.tileir_gate_calls+=1
  else:
   candidates=outputs;ret=gather_block_tables(src,counts,idx,outputs,padded);candret=ret;self.tileir_gate_calls+=1
  torch.cuda.synchronize(idx.device)
  actual=[bytecopy(t) for t in outputs];cand=[bytecopy(t) for t in candidates]
  preserved=all(torch.equal(bytecopy(t),b) for t,b in zip(inputs,before_inputs))
  exact=[torch.equal(a,c) for a,c in zip(actual,cand)]
  cpu_exact=[torch.equal(t.detach().cpu(),e) for t,e in zip(outputs,expected)]
  # Check whole backing bytes outside live writes independently, not only logical rows.
  tail=[]
  for g,(o,old,new) in enumerate(zip(outputs,before_outputs,actual)):
   mask=torch.zeros(old.numel(),dtype=torch.bool)
   for row,req in enumerate(ci.tolist()):
    n=int(cc[g,req]);start=(o.storage_offset()+row*o.stride(0))*o.element_size();mask[start:start+n*o.element_size()]=True
   tail.append(torch.equal(old[~mask],new[~mask]))
  viewok=isinstance(ret,tuple) and isinstance(candret,tuple) and len(ret)==len(candret)==4
  if viewok:
   viewok=all(same_layout(v,o[:padded]) and v.untyped_storage().data_ptr()==o.untyped_storage().data_ptr() for v,o in zip(ret,outputs)) and all(same_layout(v,o[:padded]) and v.untyped_storage().data_ptr()==o.untyped_storage().data_ptr() for v,o in zip(candret,candidates))
  passed=preserved and all(exact) and all(cpu_exact) and all(tail) and viewok
  row=dict(mode=self.mode,phase=batch['phase'],batch=batch,counts_placement=dict(buffer_class=type(active).__name__,cpu=metadata(active.cpu),accelerator=metadata(counts)),input_preserved=preserved,full_storage_raw_equal=exact,cpu_reference_exact=cpu_exact,untouched_backing_preserved=tail,views_contract=viewok,passed=passed)
  self.rows.append(row)
  if not passed and self.first_mismatch is None:
   bank=self.output/'first_mismatch';bank.mkdir();arts={}
   for name,values in [('input_before',before_inputs),('output_before',before_outputs),('native_or_actual_after',actual),('candidate_after',cand)]:
    f=bank/(name+'.pt');torch.save(values,f);arts[name]=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
   self.first_mismatch=dict(call_index=len(self.rows)-1,row=row,artifacts=arts)
  self.save();assert passed,'Integer/full-backing contract HOLD'
  return ret
 def start(self,mode):
  assert mode in ('shadow','replace') and (mode!='replace' or self.shadow_pass)
  self.mode=mode;self.rows=[];self.native_gate_calls=self.tileir_gate_calls=0;return dict(mode=mode)
 def summary(self):return dict(mode=self.mode,rows=self.rows,supported_calls=len(self.rows),native_gate_calls=self.native_gate_calls,tileir_gate_calls=self.tileir_gate_calls,shadow_pass=self.shadow_pass,first_mismatch=self.first_mismatch,supported_vendor_fallback_calls=0)
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2))
 def finish(self):
  assert len(self.rows)==16 and sum(r['phase']=='prefill' for r in self.rows)==1 and sum(r['phase']=='decode' for r in self.rows)==15
  assert all(r['passed'] for r in self.rows) and self.native_gate_calls==(16 if self.mode=='shadow' else 0) and self.tileir_gate_calls==16
  if self.mode=='shadow':self.shadow_pass=True
  self.save();return self.summary()
 def close(self):
  conflicts=[]
  for cls,name,old,new in reversed(self.bindings):
   if getattr(cls,name) is new:setattr(cls,name,old)
   else:conflicts.append(name)
  self.bindings=[];assert not conflicts,conflicts;return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR)
 from vllm.v1.worker.gpu.model_runner import GPUModelRunner
 from vllm.v1.worker.gpu.block_table import BlockTables
 a=Adapter(output)
 try:a.install_classes(GPUModelRunner,BlockTables);setattr(model,ATTR,a);return dict(boundary='prepare_attn/gather',scope='Four groups; actual32/padded32 only')
 except BaseException:a.close();raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 r=a.close();delattr(model,ATTR);return r
