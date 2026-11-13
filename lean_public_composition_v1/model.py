"""Lean model-local public API bindings; numerical implementations are imported unchanged."""
import inspect,re,types
from .support import Base,call_phase,phase_index

def layers(model,suffix):return [(n,m)for n,m in model.named_modules()if n.endswith(suffix)]
def detached(v):return v.detach() if hasattr(v,'detach')else v
class Norm(Base):
 def install(self,model):
  from tileir_vllm_gemma_norm_v3 import gemma_rms_norm
  for unsupported_name,unsupported in model.named_modules():
   if type(unsupported).__name__=='GemmaRMSNorm'and tuple(unsupported.weight.shape)!=(2560,):
    def reject(*args,_name=unsupported_name,**kwargs):raise RuntimeError('Unqualified norm '+_name)
    self.bind(unsupported,'forward',reject)
  found=[(n,m)for n,m in model.named_modules()if type(m).__name__=='GemmaRMSNorm'and tuple(m.weight.shape)==(2560,)]
  assert len(found)==65;self.owners={n for n,m in found}
  keys=[re.search(r'layers\.(\d+)\.(input_layernorm|post_attention_layernorm)$',n)for n,m in found]
  assert {(int(k[1]),k[2])for k in keys if k}=={(i,t)for i in range(32)for t in('input_layernorm','post_attention_layernorm')}and sum(k is None for k in keys)==1
  for name,m in found:
   def forward(this,x,residual=None,_name=name):
    phase=call_phase(self.counts,_name,x.shape[0]);return self.call(_name,phase,gemma_rms_norm,x.detach(),this.weight.detach(),this.variance_epsilon,detached(residual))
   self.bind(m,'forward',types.MethodType(forward,m))
  return dict(owners=sorted(self.owners))
class GatedNorm(Base):
 def install(self,model):
  from tileir_vllm_gated_norm_ordered_v1 import gated_rms_norm
  found=[(n,m)for n,m in model.named_modules()if type(m).__name__=='RMSNormGated'and tuple(m.weight.shape)==(128,)]
  assert len(found)==24;self.owners={n for n,m in found}
  assert {int(re.search(r'layers\.(\d+)\.linear_attn\.norm$',n)[1])for n,m in found}=={i for i in range(32)if i%4!=3}
  for name,m in found:
   assert m.group_size is None and m.norm_before_gate is True and m.bias is None and m.activation in('silu','swish')
   from norm_runtime_contract_capture_v1.observer import identity
   assert identity(getattr(m,'_forward_method',m.forward))==identity(m.forward_cuda)
   def forward(this,x,z,_name=name):
    assert tuple(x.shape[1:])==(32,128)
    phase=call_phase(self.counts,_name,x.shape[0]);return self.call(_name,phase,gated_rms_norm,x.detach(),z.detach(),this.weight.detach(),this.eps)
   self.bind(m,'forward',types.MethodType(forward,m))
  return dict(owners=sorted(self.owners))
class AttentionGate(Base):
 def install(self,model):
  from tileir_vllm_attention_gate_v1 import attention_sigmoid_gate
  from norm_runtime_contract_capture_v1.observer import identity
  found=layers(model,'.self_attn');assert len(found)==8;self.owners={n for n,m in found}
  for name,m in found:
   ident=identity(m.forward);assert ident['qualname']=='Qwen3NextAttention.forward' and ident['source_sha256']=='09ea2a385259edc9feadc79e304e86a31582b9770f80b8003d2d641782405818' and m.attn_output_gate is True
   def forward(this,positions,hidden_states,_name=name):
    self.guard();qkv,_=this.qkv_proj(hidden_states);q,k,v,gate=this._project_qkv_gate(qkv,positions);out=this.attn(q,k,v);assert gate is not None
    phase=call_phase(self.counts,_name,out.shape[0]);out=self.call(_name,phase,attention_sigmoid_gate,out.detach(),gate.detach());result,_=this.o_proj(out);return result
   self.bind(m,'forward',types.MethodType(forward,m))
  return dict(owners=sorted(self.owners))
class QKRope(Base):
 def install(self,model):
  import vllm.model_executor.models.qwen3_next as module
  from tileir_vllm_qk_rope_v1 import fused_qk_rmsnorm_rope_gate
  from norm_runtime_contract_capture_v1.observer import identity
  cls=module.Qwen3NextAttention;found=[(n,m)for n,m in layers(model,'.self_attn')if type(m)is cls];assert len(found)==8
  owners={id(m):n for n,m in found};self.owners=set(owners.values());original=cls._project_qkv_gate;fn=module.fused_qk_rmsnorm_rope_gate;sig=inspect.signature(fn)
  assert identity(original)['source_sha256']=='09ea2a385259edc9feadc79e304e86a31582b9770f80b8003d2d641782405818'
  assert identity(fn)['source_sha256']=='c201d88fd5e810a819cf671e8883037f53f5801211b8141e4801df1a4e5d558d'
  def project(owner,qkv,positions):
   self.guard();assert id(owner)in owners and self.pending is None and owner.use_fused_qk_norm_rope_gate is True
   ctx=dict(name=owners[id(owner)],owner=owner,qkv=qkv,positions=positions,calls=0);self.pending=ctx
   try:
    ret=original(owner,qkv,positions);assert ctx['calls']==1 and isinstance(ret,tuple)and len(ret)==4
    hr=ctx['out'];assert ret[0]is hr[0]and ret[1]is hr[1]and ret[3]is hr[2]
    v=ret[2];assert v.untyped_storage().data_ptr()==qkv.untyped_storage().data_ptr()and tuple(v.shape)==(qkv.shape[0],1024)and tuple(v.stride())==(10240,1)and v.storage_offset()==qkv.storage_offset()+9216
    return ret
   finally:self.pending=None
  def helper(*args,**kwargs):
   ctx=self.pending;assert ctx is not None,'Unscoped helper is not a lean fallback';ctx['calls']+=1;assert ctx['calls']==1
   b=sig.bind(*args,**kwargs);b.apply_defaults();v=b.arguments;m=ctx['owner']
   assert v['positions']is ctx['positions']and v['q_weight']is m.q_norm.weight and v['k_weight']is m.k_norm.weight and v['cos_sin_cache']is m.rotary_emb.cos_sin_cache
   assert v['q_gate'].untyped_storage().data_ptr()==v['k'].untyped_storage().data_ptr()==ctx['qkv'].untyped_storage().data_ptr()and tuple(v['positions'].stride())==(2049,1)
   phase=call_phase(self.counts,ctx['name'],ctx['qkv'].shape[0]);out=self.call(ctx['name'],phase,fused_qk_rmsnorm_rope_gate,**{k:detached(t)for k,t in v.items()});ctx['out']=out;return out
  self.bind(cls,'_project_qkv_gate',project);self.bind(module,'fused_qk_rmsnorm_rope_gate',helper);return dict(owners=sorted(self.owners))
class Silu(Base):
 def install(self,model):
  import torch
  from vllm.model_executor.layers.activation import SiluAndMul
  from tileir_vllm_silu_mul_v2.api import silu_and_mul
  found=layers(model,'.mlp.act_fn');assert len(found)==32;self.owners={n for n,m in found}
  for name,m in found:
   assert type(m)is SiluAndMul and m.op is torch.ops._C.silu_and_mul and getattr(m._forward_method,'__func__',None)is SiluAndMul.forward_cuda
   def op(out,x,_name=name):
    phase=call_phase(self.counts,_name,x.shape[0]);return self.call(_name,phase,silu_and_mul,out.detach(),x.detach())
   self.bind(m,'op',op)
  return dict(owners=sorted(self.owners))
class Embedding(Base):
 def install(self,model):
  from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding,UnquantizedEmbeddingMethod
  from tileir_vllm_tp1_embedding_v1 import embedding
  found=[(n,m)for n,m in layers(model,'embed_tokens')if type(m)is VocabParallelEmbedding];assert len(found)==1
  name,m=found[0];q=m.quant_method;w=m.weight;assert m.tp_size==1 and type(q)is UnquantizedEmbeddingMethod and tuple(w.shape)==(248320,2560);self.owners={name}
  def wrapper(layer,ids):
   assert layer is m and layer.weight is w and tuple(ids.shape)in((1696,),(32,))
   phase=call_phase(self.counts,name,len(ids));return self.call(name,phase,embedding,ids.detach(),w.detach())
  self.bind(q,'embedding',wrapper);return dict(owners=[name])
class Gemm(Base):
 def install(self,model):
  import torch
  from vllm.model_executor.layers.linear import UnquantizedLinearMethod
  from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
  from vllm.model_executor.layers.utils import default_unquantized_gemm,dispatch_unquantized_gemm
  from vllm import envs
  from all_gemm_runtime_integration_v1.contract import plan,phase_shape
  from all_gemm_runtime_integration_v1.adapter import metadata,dense
  from packed_k128_partial_v1 import linear as k128
  from tileirblas_once_occ2 import linear as existing
  p=plan();specs={x['path']:x for x in p['selected']};excluded={x['path']:x for x in p['excluded']};owners={n:(m,m.quant_method)for n,m in model.named_modules()if getattr(m,'quant_method',None)is not None and callable(getattr(m.quant_method,'apply',None))}
  assert set(owners)==set(specs)|set(excluded)and len({id(q)for m,q in owners.values()})==178
  assert not envs.VLLM_BATCH_INVARIANT and dispatch_unquantized_gemm()is default_unquantized_gemm
  self.owners=set(specs);self.k128_calls=0;self.existing_calls=0;self.weight_meta={}
  for name,(m,q)in owners.items():
   assert 'apply'not in vars(q)
   if name in excluded:
    assert list(m.weight.shape)==excluded[name]['weight_shape']
    def reject(this,*args,_name=name,**kwargs):self.controller.latch(self.key,RuntimeError('Excluded non-GEMM apply '+_name));raise RuntimeError('Unexpected excluded apply')
    self.bind(q,'apply',types.MethodType(reject,q));continue
   spec=specs[name];assert getattr(m,'bias',None)is None and type(q)is(UnquantizedEmbeddingMethod if spec['kind']=='head'else UnquantizedLinearMethod)
   if spec['kind']!='head':assert q._gemm_impl is default_unquantized_gemm
   dense(m.weight,spec['weight_shape'],torch);w=m.weight;wm=metadata(w);self.weight_meta[name]=wm
   def apply(this,owner,x,bias=None,_name=name,_owner=m,_q=q,_w=w,_wm=wm,_spec=spec):
    self.guard();assert this is _q and owner is _owner and owner.quant_method is _q and owner.weight is _w and bias is None
    assert metadata(owner.weight)==_wm,'Weight descriptor/version changed'
    if _spec['kind']=='head':assert not envs.VLLM_BATCH_INVARIANT and dispatch_unquantized_gemm()is default_unquantized_gemm
    else:assert this._gemm_impl is default_unquantized_gemm
    i=sum(v for(k,p),v in self.counts.items()if k==_name);phase,(M,N,K)=phase_shape(_spec,i);dense(x,[M,K],torch)
    fn=k128 if _spec['kind']=='ba'and phase=='decode'else existing
    with torch.no_grad():out=self.call(_name,phase,fn,x.detach(),_w.detach())
    if fn is k128:self.k128_calls+=1
    else:self.existing_calls+=1
    return out
   self.bind(q,'apply',types.MethodType(apply,q))
  return dict(selected_owners=153,excluded_apply_guards=25)
 def start(self):super().start();self.k128_calls=self.existing_calls=0
 def finish(self):
  r=super().finish();assert self.k128_calls==360 and self.existing_calls==2088;r.update(k128_calls=360,existing_calls=2088,weight_validation='per-call metadata/version only; no hashing in lean path');return r
