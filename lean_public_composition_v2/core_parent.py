"""Lean public prefill chain and scoped parent-state bridge. CPU-prepared, unqualified.
Numerical APIs are imported by reference. Public value/alias guards remain; no
external snapshots, hashes, native shadow/oracles or explicit synchronization.
"""
import functools
import inspect
from lean_public_composition_v1.support import Base
from norm_runtime_contract_capture_v1.observer import identity
from gdn_post_conv_runtime_capture_v1.observer import phase_meta
from gdn_parent_state_no_oracle_v2.adapter import Adapter as ParentEndpoint

NAMES=('q','k','v','g','beta','initial_state','cu_seqlens','chunk_indices','chunk_offsets')

def overlap(a,b):
 aa=a.untyped_storage();bb=b.untyped_storage()
 return aa.data_ptr()<bb.data_ptr()+bb.nbytes() and bb.data_ptr()<aa.data_ptr()+aa.nbytes()

class PrefillCore(Base):
 def install(self,model):
  from vllm.forward_context import get_forward_context
  named=dict(model.named_modules());self.core_bindings={};self.owners=set()
  for name,m in named.items():
   if not name.endswith('.linear_attn.chunk_gated_delta_rule'):continue
   owner=named[name.rsplit('.',1)[0]];original=m._forward_method;sig=inspect.signature(original)
   assert str(m.gdn_prefill_backend)=='triton' and original.__name__=='forward_native'
   self.owners.add(name)
   @functools.wraps(original)
   def wrapper(*args,_name=name,_owner=owner,_sig=sig,**kwargs):
    assert self.pending is None
    ctx=get_forward_context();assert isinstance(ctx.attn_metadata,dict)
    phase=phase_meta(ctx.attn_metadata.get(_owner.prefix))
    assert phase['phase']=='prefill' and not phase['speculative'] and phase['num_actual_tokens']==1696
    assert self.counts.get((_name,'prefill'),0)==0
    b=_sig.bind(*args,**kwargs);b.apply_defaults();self.pending=_name
    try:return self.call(_name,'prefill',self.apply,b.arguments)
    finally:self.pending=None
   self.bind(m,'_forward_method',wrapper);self.core_bindings[m]=m._forward_method
  assert len(self.owners)==len(self.core_bindings)==24
  return dict(owners=sorted(self.owners),numerical_stages=6,scope='Fresh-zero initial prefill only; unchanged public six-stage chain')
 def apply(self,v):
  import torch
  from gdn_prefill_chain_qualification_v1.chain import execute,LABELS
  from tileir_gdn_chunk_cumsum_v1 import chunk_local_cumsum
  from tileir_gdn_chunk_kkt_v5 import chunk_scaled_dot_kkt
  from tileir_gdn_solve_tril_v6 import solve_tril
  from tileir_gdn_wy_v1 import recompute_w_u_fwd
  from tileir_gdn_chunk_state_v1 import chunk_state
  from tileir_gdn_chunk_output_v2 import chunk_output
  from tileir_gdn_chunk_cumsum_v1.contract import validate_metadata
  assert {k:v[k] for k in v if k not in NAMES}==dict(output_final_state=True,use_qk_l2norm_in_kernel=False,core_attn_out=None,aiter_prefill_metadata=None)
  inputs={k:v[k].detach() for k in NAMES}
  specs={'q':((1,1696,16,128),torch.bfloat16),'k':((1,1696,16,128),torch.bfloat16),'v':((1,1696,32,128),torch.bfloat16),'g':((1,1696,32),torch.float32),'beta':((1,1696,32),torch.float32),'initial_state':((32,32,128,128),torch.float32),'cu_seqlens':((33,),torch.int32),'chunk_indices':((32,2),torch.int32),'chunk_offsets':((33,),torch.int64)}
  for k,t in inputs.items():
   shape,dtype=specs[k]
   assert t.is_cuda and t.device==inputs['q'].device and tuple(t.shape)==shape and t.dtype==dtype and t.is_contiguous() and t.storage_offset()==0 and t.untyped_storage().nbytes()==t.numel()*t.element_size()<=128*1024*1024
  validate_metadata(inputs['cu_seqlens'].cpu().tolist(),inputs['chunk_indices'].cpu().tolist())
  assert inputs['chunk_offsets'].cpu().tolist()==list(range(33))
  assert all(bool(torch.isfinite(t).all()) for t in inputs.values())
  assert bool((inputs['initial_state']==0).all()),'Nonzero initial state outside qualified scope'
  ops=dict(cumsum=chunk_local_cumsum,kkt=chunk_scaled_dot_kkt,solve=solve_tril,wy=recompute_w_u_fwd,state=chunk_state,output=chunk_output)
  retained={}
  def observe(outputs):
   for label,t in outputs.items():
    assert label not in retained and bool(torch.isfinite(t).all()) and t.is_contiguous() and t.storage_offset()==0 and t.untyped_storage().nbytes()==t.numel()*t.element_size()<=128*1024*1024
    assert all(not overlap(t,z) for z in [*inputs.values(),*retained.values()]);retained[label]=t
  with torch.inference_mode():ret=execute(inputs,ops,observe)
  assert list(retained)==list(LABELS)
  assert isinstance(ret,tuple) and len(ret)==2
  assert ret[0].dtype==torch.bfloat16 and tuple(ret[0].shape)==(1,1696,32,128)
  assert ret[1].dtype==torch.float32 and tuple(ret[1].shape)==(32,32,128,128)
  return ret
 def finish(self):
  r=super().finish();assert dict(self.counts)=={(n,'prefill'):1 for n in self.owners}
  return r

class _LeanParentEndpoint(ParentEndpoint):
 """Inherited exact scoped owner dispatch with no file I/O and no native target arm."""
 def __init__(self,part):
  # Do not call the diagnostic constructor: no output path/files are owned.
  self.part=part;self.bindings=[];self.modules={};self.mode='replace';self.rows=[]
  self.failed=None;self.active=None;self.baseline_pass=False
 def guard(self):
  c=self.part.controller
  assert c.active and c.failed is None and self.failed is None,'Lean parent outside request or latched'
 def fail(self,error):
  if self.failed is None:self.failed=dict(error=repr(error))
  self.part.controller.latch(self.part.key,error)
 def save(self):
  # No diagnostic serialization or value snapshots; rows retain scope/order metadata.
  return dict(calls=len(self.rows),target_oracles=0)
 def install(self,model):
  # ParentEndpoint.install invokes no tensors; its callable identity is naturally
  # inherited through functools.wraps, never forged to mimic composition guards.
  return super().install(model)

class GDNParent(Base):
 def install(self,model):
  matches=[p for p in self.controller.parts if p.key=='gdn_prefill_core']
  assert len(matches)==1;self.core=matches[0]
  assert isinstance(self.core,PrefillCore) and len(self.core.core_bindings)==24
  self.expected_core=dict(self.core.core_bindings)
  assert all(m._forward_method is fn for m,fn in self.expected_core.items())
  self.endpoint=_LeanParentEndpoint(self)
  try:
   r=self.endpoint.install(model)
   seen={}
   for obj,key,had,old,new in self.endpoint.bindings:
    if key!='_forward_method':continue
    assert obj in self.expected_core and had and old is self.expected_core[obj]
    assert new.__wrapped__ is old and obj._forward_method is new
    seen[obj]=new
   assert set(seen)==set(self.expected_core)
  except BaseException:
   self.endpoint.close();raise
  return dict(owners=sorted(self.endpoint.modules),core_delegated=True,core_native=False,logical_operations=96,target_oracles=0)
 def start(self):
  super().start();assert self.endpoint.active is None and self.endpoint.failed is None
  self.endpoint.rows=[];self.endpoint.mode='replace'
 def finish(self):
  e=self.endpoint;e.guard();assert e.active is None
  assert len(e.rows)==24 and {r['layer'] for r in e.rows}==set(e.modules)
  for r in e.rows:
   assert r['passed'] and r['order']==['gather','invert','mask','core_enter','core_exit','scatter']
   assert r['native_target_calls']==r['native_invert_calls']==0
   assert r['tile_target_calls']==3 and r['tile_invert_calls']==1
  self.counts.update({(n,'prefill'):1 for n in e.modules})
  r=super().finish();r.update(logical_suboperation_calls=96,parent_api_calls=96,core_delegated=True,target_oracles=0)
  return r
 def close(self):
  if not hasattr(self,'endpoint'):return dict(restored=True,not_installed=True)
  r=self.endpoint.close()
  assert all(m._forward_method is fn for m,fn in self.expected_core.items()),'Parent must restore actual lean core first'
  r['core_binding_restored']=True
  return r
