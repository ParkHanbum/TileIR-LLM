"""Preserve live mutation/alias/state owners; omit only proof shadows and comparisons."""
import functools,inspect
from .support import Base
from .model import detached

def gdn_owners(model):
 found={n:m for n,m in model.named_modules()if n.endswith('.linear_attn')and hasattr(m,'A_log')and hasattr(m,'dt_bias')}
 assert len(found)==24;return found
class PrefillConv(Base):
 def install(self,model):
  import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q
  from tileir_indexed_prefill_conv_v1 import causal_conv1d_prefill_indexed
  live=gdn_owners(model);byw={m.conv1d.weight.data_ptr():n for n,m in live.items()};assert len(byw)==24;self.owners=set(live);self.update_at_install=(q,q.causal_conv1d_update)
  orig=q.causal_conv1d_fn;sig=inspect.signature(orig)
  @functools.wraps(orig)
  def wrapper(*args,**kwargs):
   bound=sig.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments;name=byw.get(v['weight'].data_ptr());assert name is not None
   assert v['bias']is None and v['activation']=='silu'and v['null_block_id']==0 and v['pad_slot_id']==-1 and v['block_size_to_align']==0
   assert all(v[k]is None for k in('block_idx_first_scheduled_token','block_idx_last_scheduled_token','initial_state_idx','num_computed_tokens'))
   x,st,w,idx,starts,flags=(v[k]for k in('x','conv_states','weight','cache_indices','query_start_loc','has_initial_state'))
   assert tuple(x.shape)==(8192,1696)and tuple(st.shape[1:])==(8192,3)
   # Scope guards, not output comparisons; retain original no-initial-state53-token domain.
   assert starts.detach().cpu().tolist()==list(range(0,1697,53))and not any(flags.detach().cpu().tolist())
   assert self.counts.get((name,'prefill'),0)==0
   return self.call(name,'prefill',causal_conv1d_prefill_indexed,x.detach(),w.detach(),st.detach(),starts,idx,flags)
  self.bind(q,'causal_conv1d_fn',wrapper);return dict(owners=sorted(self.owners))
 def close(self):
  r=super().close();r['update_binding_is_install_time_object']=self.update_at_install[0].causal_conv1d_update is self.update_at_install[1];assert r['update_binding_is_install_time_object'];return r
class DecodeConv(Base):
 def install(self,model):
  import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q
  from tileir_indexed_decode_conv_v1 import causal_conv1d_update_indexed
  live=gdn_owners(model);byw={m.conv1d.weight.data_ptr():n for n,m in live.items()};self.owners=set(live);self.delegated_prefill=0
  for api,phase in [('causal_conv1d_fn','prefill'),('causal_conv1d_update','decode')]:
   orig=getattr(q,api);sig=inspect.signature(orig)
   @functools.wraps(orig)
   def wrapper(*args,_orig=orig,_sig=sig,_phase=phase,**kwargs):
    bound=_sig.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments;name=byw.get(v['weight'].data_ptr());assert name is not None
    if _phase=='prefill':self.delegated_prefill+=1;assert self.delegated_prefill<=24;return _orig(*args,**kwargs) # already installed TileIR prefill wrapper
    assert v['bias']is None and v['out']is None and v['activation']=='silu'and v['null_block_id']==0 and v['max_query_len']==-1
    assert all(v[k]is None for k in('num_accepted_tokens','query_start_loc','block_idx_last_scheduled_token','initial_state_idx'))
    x,st,w,idx=v['x'],v['conv_state'],v['weight'],v['conv_state_indices'];assert tuple(x.shape)==(32,8192)and tuple(st.shape[1:])==(8192,3)and self.counts.get((name,'decode'),0)<15
    # API mutates original X/state through detached aliases, returns distinct same-storage view.
    return self.call(name,'decode',causal_conv1d_update_indexed,x.detach(),st.detach(),w.detach(),idx.detach())
   self.bind(q,api,wrapper)
  return dict(owners=sorted(self.owners))
 def start(self):super().start();self.delegated_prefill=0
 def finish(self):r=super().finish();assert self.delegated_prefill==24;r['delegated_prefill_tileir']=24;return r
class PostConv(Base):
 def install(self,model):
  import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q
  from vllm.forward_context import get_forward_context
  from tileir_vllm_post_conv_v2.api import fused_post_conv_prep
  from gdn_post_conv_runtime_capture_v1.observer import phase_meta
  live=gdn_owners(model);self.owners=set(live);orig=q.fused_post_conv_prep;sig=inspect.signature(orig)
  @functools.wraps(orig)
  def wrapper(*args,**kwargs):
   bound=sig.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments;found=[(n,m)for n,m in live.items()if v['A_log']is m.A_log and v['dt_bias']is m.dt_bias];assert len(found)==1;name,m=found[0]
   ctx=get_forward_context();assert isinstance(ctx.attn_metadata,dict);phase=phase_meta(ctx.attn_metadata.get(m.prefix));assert phase['phase']=='prefill'and not phase['speculative']and self.counts.get((name,'prefill'),0)==0
   d=dict(v)
   import torch
   for k in('conv_output','a','b','A_log','dt_bias'):
    assert bool(torch.isfinite(v[k]).all()) # original finite-input scope
    d[k]=v[k].detach()
   return self.call(name,'prefill',fused_post_conv_prep,**d)
  self.bind(q,'fused_post_conv_prep',wrapper);return dict(owners=sorted(self.owners))
class GDNDecode(Base):
 def install(self,model):
  import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q
  from tileir_gdn_decode_core_v4 import packed_decode
  live=gdn_owners(model);owners={(m.A_log.data_ptr(),m.dt_bias.data_ptr()):n for n,m in live.items()};assert len(owners)==24;self.owners=set(live)
  self.untouched={k:getattr(q,k)for k in('causal_conv1d_fn','causal_conv1d_update','fused_post_conv_prep')};self.q=q
  key='fused_recurrent_gated_delta_rule_packed_decode';orig=getattr(q,key);sig=inspect.signature(orig)
  @functools.wraps(orig)
  def wrapper(*args,**kwargs):
   bound=sig.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments;name=owners.get((v['A_log'].data_ptr(),v['dt_bias'].data_ptr()));assert name is not None and self.counts.get((name,'decode'),0)<15
   st=v['initial_state'];out=v['out'];assert tuple(st.shape[1:])==(32,128,128)and tuple(st.stride()[1:])==(16384,128,1)
   assert v['scale']==128**-0.5 and v['use_qk_l2norm_in_kernel']is True
   # Preserve observed positive unique32 scope, beyond public API's broader null allowance.
   slots=v['ssm_state_indices'].detach().cpu().tolist();assert len(slots)==len(set(slots))==32 and all(0<s<st.shape[0]for s in slots)
   d={k:detached(t)for k,t in v.items()};got=self.call(name,'decode',packed_decode,**d)
   assert isinstance(got,tuple)and len(got)==2 and got[0]is d['out']and got[1]is d['initial_state']
   return out,st # original caller identities, not detached aliases
  self.bind(q,key,wrapper);return dict(owners=sorted(self.owners))
 def close(self):r=super().close();assert all(getattr(self.q,k)is v for k,v in self.untouched.items());return r
