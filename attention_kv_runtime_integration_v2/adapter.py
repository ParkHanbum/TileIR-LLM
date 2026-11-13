"""Actual shared-cache insertion adapter; shadow candidate writes private cache only."""
from pathlib import Path
import sys,json,inspect
from collections import Counter
from attention_kv_runtime_capture_v1.observer import Observer,phase_from_lists,metadata,identity
ATTR='_tileir_attention_kv_integration_v1'
class Adapter(Observer):
 def __init__(self,out):
  super().__init__(out);self.mode='shadow';self.shadow_pass=False;self.rows=[];self.native_gate_calls=0;self.tileir_gate_calls=0
 def install(self,model):
  from vllm.forward_context import get_forward_context
  found=[(n,m) for n,m in model.named_modules() if n.endswith('.self_attn.attn')]
  assert len(found)==8 and len({id(m.impl) for n,m in found})==8
  helpers={}
  for name,owner in found:
   impl=owner.impl;original=impl.do_kv_cache_update;mod=sys.modules[original.__module__]
   self.modules[name]=dict(implementation=type(impl).__module__+'.'+type(impl).__name__,method=identity(original),cache_dtype=str(impl.kv_cache_dtype))
   self.progress()
   assert type(impl).__name__=='FlashAttentionImpl' and hasattr(mod,'reshape_and_cache_flash'),'Unsupported actual backend; not forced'
   helpers[id(mod)]=mod
   def update(*args,_name=name,_owner=owner,_original=original,**kwargs):
    assert self.pending is None
    b=inspect.signature(_original).bind(*args,**kwargs);b.apply_defaults();v=b.arguments
    assert v['layer'] is _owner and v['kv_cache'] is _owner.kv_cache
    ctx=get_forward_context();assert isinstance(ctx.attn_metadata,dict);md=ctx.attn_metadata.get(_owner.layer_name)
    starts=md.query_start_loc.detach().cpu().tolist();lengths=md.seq_lens.detach().cpu().tolist();phase=phase_from_lists(starts,lengths)
    context=dict(name=_name,phase=phase,owner=_owner,args=v,metadata=dict(query_start_loc=starts,seq_lens=lengths),calls=0)
    self.pending=context
    try:
     ret=_original(*args,**kwargs);assert context['calls']==1 and ret is None
     self.counts[(_name,phase)]+=1;self.progress();return ret
    finally:self.pending=None
   had='do_kv_cache_update' in impl.__dict__;prior=impl.__dict__.get('do_kv_cache_update');self.bindings.append((impl,'do_kv_cache_update',had,prior,update));impl.do_kv_cache_update=update
  for mod in helpers.values():
   original=mod.reshape_and_cache_flash;sig=inspect.signature(original)
   def helper(*args,_original=original,_sig=sig,**kwargs):
    ctx=self.pending
    if ctx is None:return _original(*args,**kwargs)
    ctx['calls']+=1;assert ctx['calls']==1
    b=_sig.bind(*args,**kwargs);b.apply_defaults();v=b.arguments
    assert v['key'] is ctx['args']['key'] and v['value'] is ctx['args']['value'] and v['slot_mapping'] is ctx['args']['slot_mapping']
    return self.apply(ctx,v,lambda:_original(*args,**kwargs))
   self.bindings.append((mod,'reshape_and_cache_flash',True,original,helper));mod.reshape_and_cache_flash=helper
  return self.progress()
 def apply(self,ctx,v,native):
  import torch
  from tileir_vllm_kv_scatter_v1.api import reshape_and_cache_flash,validate
  from gdn_post_conv_qualification_v2.run import compare
  from attention_kv_qualification_v2.run import selected_gate
  vals={k:v[k] for k in ('key','value','slot_mapping','k_scale','v_scale')};args=dict(v);detach={}
  for k,t in vals.items():
   z=t.detach();args[k]=z;detach[k]=dict(original_requires_grad=bool(t.requires_grad),same_storage=z.untyped_storage().data_ptr()==t.untyped_storage().data_ptr(),same_stride=z.stride()==t.stride(),same_offset=z.storage_offset()==t.storage_offset());assert all(detach[k][n] for n in ('same_storage','same_stride','same_offset'))
  validate(**args)
  def raw(t):return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
  before={k:raw(t) for k,t in vals.items()};assert all(bool(torch.isfinite(v[k]).all()) for k in ('key','value'))
  kc,vc=v['key_cache'],v['value_cache'];slots=v['slot_mapping'].to(torch.int64);bs=kc.shape[1];blocks=slots//bs;offs=slots%bs
  metrics={};private_bytes=0;none=True
  if self.mode=='shadow':
   ret=native();self.native_gate_calls+=1;assert ret is None
   nb=int(blocks.max())+1;shape=(nb,)+tuple(kc.shape[1:]);descs=[(t.stride(),t.storage_offset()-kc.storage_offset()) for t in (kc,vc)]
   extent=max(off+sum((n-1)*st for n,st in zip(shape,stride))+1 for stride,off in descs);private_bytes=extent*2;assert private_bytes<=384*1024*1024
   backing=torch.full((extent,),-37,dtype=torch.bfloat16,device=kc.device);pk,pv=[torch.as_strided(backing,shape,st,off) for st,off in descs]
   private=dict(args,key_cache=pk,value_cache=pv);cr=reshape_and_cache_flash(**private);self.tileir_gate_calls+=1;none=cr is None
   for name,a,b in [('key',kc,pk),('value',vc,pv)]:metrics[name]=compare(a[blocks,offs].cpu(),b[blocks,offs].cpu(),torch)
   # Full private sentinel check, never clone or hash engine shared backing.
   expected=torch.full((extent,),-37,dtype=torch.bfloat16);ek,ev=[torch.as_strided(expected,shape,st,off) for st,off in descs]
   cb=blocks.cpu();co=offs.cpu();ek[cb,co]=kc[blocks,offs].cpu();ev[cb,co]=vc[blocks,offs].cpu()
   sentinel=torch.equal(raw(backing),raw(expected));passed=selected_gate(metrics) and sentinel and none
  else:
   assert self.mode=='replace' and self.shadow_pass
   ret=reshape_and_cache_flash(**args);self.tileir_gate_calls+=1;none=ret is None
   for name,cache in [('key',kc),('value',vc)]:metrics[name]=compare(v[name][:len(slots)].detach().cpu(),cache[blocks,offs].cpu(),torch)
   sentinel=None;passed=selected_gate(metrics) and none
  preserved=all(torch.equal(before[k],raw(t)) for k,t in vals.items());passed &= preserved
  row=dict(path=ctx['name'],phase=ctx['phase'],mode=self.mode,passed=passed,metrics=metrics,input_preserved=preserved,return_none=none,private_sentinel_pass=sentinel,private_bytes=private_bytes,private_cache_offset_rebase=kc.storage_offset(),actual_cache_unselected_verified=False,detach=detach,cache_descriptors={'key':metadata(kc),'value':metadata(vc)},phase_evidence=ctx['metadata'])
  self.rows.append(row);self.progress()
  if not passed:
   bank=self.out/'first_mismatch';bank.mkdir(exist_ok=False)
   for name,t in dict(vals,actual_selected_key=kc[blocks,offs],actual_selected_value=vc[blocks,offs]).items():torch.save(t.detach().cpu(),bank/(name+'.pt'))
   raise AssertionError('KV raw/strict/finite/storage HOLD')
  return ret
 def start(self,mode):
  assert mode in ('shadow','replace') and (mode!='replace' or self.shadow_pass)
  self.mode=mode;self.rows=[];self.counts=Counter();self.native_gate_calls=self.tileir_gate_calls=0;return dict(mode=mode)
 def progress(self):
  r=dict(mode=self.mode,rows=self.rows,supported_calls=len(self.rows),native_gate_calls=self.native_gate_calls,tileir_gate_calls=self.tileir_gate_calls,shadow_pass=self.shadow_pass,supported_vendor_fallback_calls=0)
  (self.out/(self.mode+'-PROGRESS.json')).write_text(json.dumps(r,indent=2));return r
 def finish(self):
  assert self.pending is None and self.counts==Counter({(n,p):c for n in self.modules for p,c in [('prefill',1),('decode',15)]})
  assert len(self.rows)==128 and all(x['passed'] for x in self.rows) and self.tileir_gate_calls==128 and self.native_gate_calls==(128 if self.mode=='shadow' else 0)
  if self.mode=='shadow':self.shadow_pass=True
  return self.progress()
 def close(self):super().close();return dict(restored=True)
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
