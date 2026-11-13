"""Only public compute_slot_mappings boundary replaced after full native shadow."""
from pathlib import Path
import json,hashlib
from block_gather_runtime_capture_v1.observer import phase_metadata,metadata
from block_gather_runtime_integration_v1.adapter import raw,bytecopy,clone_storage,same_layout
ATTR='_tileir_slot_mapping_integration_v1'
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.bindings=[];self.pending=None;self.mode='shadow';self.rows=[];self.shadow_pass=False;self.native_gate_calls=0;self.tileir_gate_calls=0;self.first_mismatch=None
 def install_classes(self,R,B):
  rp=R.prepare_attn;sm=B.compute_slot_mappings
  def prepare(r,b,*args,**kwargs):
   assert self.pending is None and r.pcp_manager is None
   ctx=dict(owner=r.block_tables,batch=b,metadata=phase_metadata(b),calls=0);self.pending=ctx
   try:
    ret=rp(r,b,*args,**kwargs);assert ctx['calls']==1;return ret
   finally:self.pending=None
  def slots(owner,idx_mapping,query_start_loc,positions,num_tokens_padded,out=None):
   ctx=self.pending;assert ctx is not None and owner is ctx['owner'];ctx['calls']+=1;b=ctx['batch']
   assert out is None and idx_mapping is b.idx_mapping and query_start_loc is b.query_start_loc and positions is b.positions and num_tokens_padded==b.num_tokens_after_padding
   return self.apply(owner,idx_mapping,query_start_loc,positions,num_tokens_padded,lambda:sm(owner,idx_mapping,query_start_loc,positions,num_tokens_padded,out=out),ctx['metadata'])
  for cls,name,old,new in ((R,'prepare_attn',rp,prepare),(B,'compute_slot_mappings',sm,slots)):
   self.bindings.append((cls,name,old,new));setattr(cls,name,new)
 def apply(self,owner,idx,starts,pos,padded,native,batch):
  import torch
  from tileir_vllm_slot_mapping_v1.api import compute_slot_mappings,reference_cpu
  assert self.mode in ('shadow','replace') and (self.mode!='replace' or self.shadow_pass)
  assert owner.cp_size==1 and batch['num_reqs']==32 and batch['phase'] in ('prefill','decode')
  src=tuple(t.gpu for t in owner.block_tables);sizes=owner.block_sizes_tensor;enabled=owner.slot_mapping_enabled;output=owner.slot_mappings
  assert [tuple(t.shape) for t in src]==[(32,1)]*3+[(32,8)] and tuple(output.shape)==(4,2048)
  inputs=(*src,idx,starts,pos,sizes,enabled,owner.dcp_sharded,owner.block_table_ptrs,owner.block_table_strides)
  before=[bytecopy(t) for t in inputs];oldout=bytecopy(output)
  cpu=[t.detach().cpu().clone() for t in (*src,idx,starts,pos,sizes,enabled)]
  assert cpu[7].tolist()==[128,128,128,528] and bool(cpu[8].all())
  expected=output.detach().cpu().clone();reference_cpu(tuple(cpu[:4]),*cpu[4:],expected,padded);actual_tokens=int(cpu[5][-1])
  if self.mode=='shadow':
   candidate=clone_storage(output)
   ret=native();self.native_gate_calls+=1
   candret=compute_slot_mappings(src,idx,starts,pos,sizes,enabled,candidate,padded,cp_size=1);self.tileir_gate_calls+=1
  else:
   candidate=output;ret=compute_slot_mappings(src,idx,starts,pos,sizes,enabled,output,padded,cp_size=1);candret=ret;self.tileir_gate_calls+=1
  torch.cuda.synchronize(output.device)
  actual=bytecopy(output);cand=bytecopy(candidate)
  preserved=all(torch.equal(bytecopy(t),b) for t,b in zip(inputs,before));exact=torch.equal(actual,cand);cpuexact=torch.equal(output.detach().cpu(),expected)
  alias=same_layout(ret,output[:,:padded]) and ret.untyped_storage().data_ptr()==output.untyped_storage().data_ptr() and same_layout(candret,candidate[:,:padded]) and candret.untyped_storage().data_ptr()==candidate.untyped_storage().data_ptr()
  pad=bool((output[:,actual_tokens:]==-1).all().item());passed=preserved and exact and cpuexact and alias and pad
  row=dict(mode=self.mode,phase=batch['phase'],batch=batch,original_input_descriptors=[metadata(t) for t in inputs],input_preserved=preserved,full_storage_raw_equal=exact,cpu_reference_exact=cpuexact,full_capacity_pad=pad,views_contract=alias,return_shape=list(ret.shape),return_stride=list(ret.stride()),passed=passed)
  self.rows.append(row)
  if not passed and self.first_mismatch is None:
   bank=self.output/'first_mismatch';bank.mkdir();art={}
   for name,value in [('input_before',before),('output_before',oldout),('native_or_actual_after',actual),('candidate_after',cand)]:
    f=bank/(name+'.pt');torch.save(value,f);art[name]=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
   self.first_mismatch=dict(call_index=len(self.rows)-1,row=row,artifacts=art)
  self.save();assert passed,'Slot raw/full-capacity contract HOLD';return ret
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
  for cls,name,old,new in reversed(self.bindings):
   if getattr(cls,name) is new:setattr(cls,name,old)
   else:conflicts.append(name)
  self.bindings=[];assert not conflicts,conflicts;return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR)
 from vllm.v1.worker.gpu.model_runner import GPUModelRunner
 from vllm.v1.worker.gpu.block_table import BlockTables
 a=Adapter(output)
 try:a.install_classes(GPUModelRunner,BlockTables);setattr(model,ATTR,a);return dict(boundary='prepare_attn/compute_slot_mappings',scope='CP1/all4enabled/fullcapacity2048')
 except BaseException:a.close();raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 r=a.close();delattr(model,ATTR);return r
