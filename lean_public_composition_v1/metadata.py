"""Lean runner metadata/KV bindings. Scope guards are kept; no full-backing oracle copies."""
import inspect
from .support import Base
from block_gather_runtime_capture_v1.observer import phase_metadata
class Gather(Base):
 def install(self,model):
  from vllm.v1.worker.gpu.model_runner import GPUModelRunner
  from vllm.v1.worker.gpu.block_table import BlockTables
  self.install_classes(GPUModelRunner,BlockTables);return dict(boundary='prepare_attn/gather')
 def install_classes(self,R,B):
  original=R.prepare_attn
  def prepare(owner,batch,*args,**kwargs):
   assert self.pending is None and owner.pcp_manager is None
   self.pending=dict(owner=owner.block_tables,batch=batch,metadata=phase_metadata(batch),calls=0)
   try:ret=original(owner,batch,*args,**kwargs);assert self.pending['calls']==1;return ret
   finally:self.pending=None
  def gather(owner,idx,num_reqs_padded,out=None,out_ptrs=None):
   indices,padded=idx,num_reqs_padded
   ctx=self.pending;assert ctx is not None and owner is ctx['owner']and indices is ctx['batch'].idx_mapping and out is out_ptrs is None
   ctx['calls']+=1;assert ctx['calls']==1
   return self.apply(owner,indices,padded,ctx['metadata'])
  self.bind(R,'prepare_attn',prepare);self.bind(B,'gather_block_tables',gather)
 def apply(self,owner,indices,padded,batch):
  import torch
  from tileir_vllm_block_gather_v1.api import gather_block_tables
  assert batch['num_reqs']==batch['num_reqs_after_padding']==padded==32 and batch['phase']in('prefill','decode')
  src=tuple(t.gpu for t in owner.block_tables);out=tuple(owner.input_block_tables);counts=owner.num_blocks.gpu
  assert [tuple(t.shape)for t in src]==[(32,1)]*3+[(32,8)]and indices.dtype==torch.int64 and tuple(counts.shape)==(4,32)
  pool=owner.num_blocks.pool;active=pool._uva_bufs[pool._curr]
  assert type(active).__name__=='UvaBuffer'and active.cpu.is_pinned()and counts.data_ptr()==active.cpu.data_ptr()==active._uva.data_ptr()
  self.controller.owner_refs.extend((owner,pool,active,active.cpu,active._uva,counts))
  return self.call('runner',batch['phase'],gather_block_tables,src,counts,indices,out,padded)
class Slots(Base):
 def install(self,model):
  from vllm.v1.worker.gpu.model_runner import GPUModelRunner
  from vllm.v1.worker.gpu.block_table import BlockTables
  self.install_classes(GPUModelRunner,BlockTables);return dict(boundary='prepare_attn/slot_mapping')
 def install_classes(self,R,B):
  original=R.prepare_attn
  def prepare(owner,batch,*args,**kwargs):
   assert self.pending is None and owner.pcp_manager is None
   self.pending=dict(owner=owner.block_tables,batch=batch,metadata=phase_metadata(batch),calls=0)
   try:ret=original(owner,batch,*args,**kwargs);assert self.pending['calls']==1;return ret
   finally:self.pending=None
  def slots(owner,idx_mapping,query_start_loc,positions,num_tokens_padded,out=None):
   indices,starts,padded=idx_mapping,query_start_loc,num_tokens_padded
   ctx=self.pending;assert ctx is not None and owner is ctx['owner'];b=ctx['batch'];ctx['calls']+=1
   assert ctx['calls']==1 and out is None and indices is b.idx_mapping and starts is b.query_start_loc and positions is b.positions and padded==b.num_tokens_after_padding
   return self.apply(owner,indices,starts,positions,padded,ctx['metadata'])
  self.bind(R,'prepare_attn',prepare);self.bind(B,'compute_slot_mappings',slots)
 def apply(self,owner,indices,starts,positions,padded,batch):
  from tileir_vllm_slot_mapping_v1.api import compute_slot_mappings
  assert owner.cp_size==1 and batch['num_reqs']==32 and batch['phase']in('prefill','decode')
  src=tuple(t.gpu for t in owner.block_tables);sizes=owner.block_sizes_tensor;enabled=owner.slot_mapping_enabled;out=owner.slot_mappings
  assert [tuple(t.shape)for t in src]==[(32,1)]*3+[(32,8)]and tuple(out.shape)==(4,2048)
  # These two small value guards define the qualified all-enabled block-size scope.
  assert sizes.detach().cpu().tolist()==[128,128,128,528]and all(enabled.detach().cpu().tolist())
  return self.call('runner',batch['phase'],compute_slot_mappings,src,indices,starts,positions,sizes,enabled,out,padded,cp_size=1)
class Positions(Base):
 def install(self,model):
  import vllm.v1.worker.gpu.model_runner as module
  self.install_classes(module.GPUModelRunner,module);return dict(boundary='prepare_inputs/prepare_pos_seq_lens')
 def install_classes(self,R,module):
  from pos_seq_runtime_capture_v2.observer import state_metadata
  original=R.prepare_inputs
  def prepare(owner,scheduler_output,batch_req_state,batch_desc,num_active_loras):
   assert self.pending is None and owner.pcp_manager is None
   self.pending=dict(owner=owner,batch=state_metadata(batch_req_state),calls=0)
   try:ret=original(owner,scheduler_output,batch_req_state,batch_desc,num_active_loras);assert self.pending['calls']==1;return ret
   finally:self.pending=None
  def helper(idx_mapping,query_start_loc,num_computed_tokens,pos,seq_lens):
   indices,starts,computed,seq=idx_mapping,query_start_loc,num_computed_tokens,seq_lens
   from tileir_vllm_pos_seq_v1 import prepare_pos_seq_lens
   ctx=self.pending;assert ctx is not None;owner=ctx['owner'];ctx['calls']+=1;assert ctx['calls']==1 and ctx['batch']['phase']in('prefill','decode')
   assert pos is owner.input_buffers.positions and seq is owner.input_buffers.seq_lens and computed is owner.req_states.num_computed_tokens.gpu and not hasattr(owner.req_states.num_computed_tokens,'_uva_buf')
   ids=indices.detach().cpu().tolist();prefix=starts.detach().cpu().tolist();counts=computed.detach().cpu().tolist()
   assert ids==ctx['batch']['idx_mapping'] and 0<=len(ids)<=len(seq)<=32 and len(prefix)>=len(ids)+1
   assert all(0<=i<len(counts)for i in ids)and prefix[0]>=0 and prefix[len(ids)]<=len(pos)and all(a<=b for a,b in zip(prefix[:len(ids)],prefix[1:len(ids)+1]))
   assert all(0<=counts[i]and counts[i]+prefix[j+1]-prefix[j]<=2147483647 for j,i in enumerate(ids))
   return self.call('runner',ctx['batch']['phase'],prepare_pos_seq_lens,indices,starts,computed,pos,seq)
  self.bind(R,'prepare_inputs',prepare);self.bind(module,'prepare_pos_seq_lens',helper)
class KV(Base):
 def install(self,model):
  import sys
  from vllm.forward_context import get_forward_context
  from attention_kv_runtime_capture_v1.observer import phase_from_lists
  found=[(n,m)for n,m in model.named_modules()if n.endswith('.self_attn.attn')];assert len(found)==8 and len({id(m.impl)for n,m in found})==8
  self.owners={n for n,m in found};modules={}
  for name,m in found:
   impl=m.impl;orig=impl.do_kv_cache_update;mod=sys.modules[orig.__module__];assert type(impl).__name__=='FlashAttentionImpl'and hasattr(mod,'reshape_and_cache_flash');modules[id(mod)]=mod;sig=inspect.signature(orig)
   def update(*args,_name=name,_owner=m,_orig=orig,_sig=sig,**kwargs):
    assert self.pending is None
    b=_sig.bind(*args,**kwargs);b.apply_defaults();v=b.arguments;assert v['layer']is _owner and v['kv_cache']is _owner.kv_cache
    ctx=get_forward_context();assert isinstance(ctx.attn_metadata,dict);md=ctx.attn_metadata.get(_owner.layer_name)
    phase=phase_from_lists(md.query_start_loc.detach().cpu().tolist(),md.seq_lens.detach().cpu().tolist());self.pending=dict(name=_name,phase=phase,values=v,calls=0)
    try:ret=_orig(*args,**kwargs);assert self.pending['calls']==1 and ret is None;return ret
    finally:self.pending=None
   self.bind(impl,'do_kv_cache_update',update)
  for mod in modules.values():
   sig=inspect.signature(mod.reshape_and_cache_flash)
   def helper(*args,_sig=sig,**kwargs):
    ctx=self.pending;assert ctx is not None,'No unscoped KV fallback';ctx['calls']+=1;assert ctx['calls']==1
    b=_sig.bind(*args,**kwargs);b.apply_defaults();v=b.arguments
    assert all(v[k]is ctx['values'][k]for k in('key','value','slot_mapping'))
    return self.apply(ctx,v)
   self.bind(mod,'reshape_and_cache_flash',helper)
  return dict(owners=sorted(self.owners))
 def apply(self,ctx,v):
  import torch
  from tileir_vllm_kv_scatter_v1.api import reshape_and_cache_flash
  d=dict(v)
  for k in('key','value','slot_mapping','k_scale','v_scale'):d[k]=v[k].detach()
  assert all(bool(torch.isfinite(v[k]).all())for k in('key','value')) # original finite-input scope
  return self.call(ctx['name'],ctx['phase'],reshape_and_cache_flash,**d)
