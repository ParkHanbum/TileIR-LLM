"""Public FlashAttention helper observation; native-only, no implementation replacement."""
from pathlib import Path
import sys,json,hashlib,inspect,functools
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from norm_runtime_contract_capture_v1.observer import metadata as gpu_metadata,identity,aliases
from attention_kv_runtime_capture_v1.observer import phase_from_lists
ATTR='_attention_core_native_v1'
CAP=256*1024*1024
def metadata(t):
 if t.device.type=='cuda':return gpu_metadata(t)
 return dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),dtype=str(t.dtype),device=str(t.device),storage_nbytes=t.untyped_storage().nbytes(),data_ptr=t.data_ptr(),storage_ptr=t.untyped_storage().data_ptr(),object_identity=id(t),stream=None)
class Observer:
 def __init__(self,out):
  self.out=Path(out);self.out.mkdir(exist_ok=False);self.bindings=[];self.modules={};self.counts=Counter();self.samples=[];self.pending=None;self.bytes=0;self.calls=[]
 def digest(self,t):
  import torch
  return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 def tensor(self,t,label):
  import torch
  n=t.numel()*t.element_size();assert self.bytes+n<=CAP
  p=self.out/(label+'.pt');assert not p.exists();c=t.detach().cpu().contiguous();torch.save(c,p);self.bytes+=n
  return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),logical_sha256=self.digest(c),metadata=metadata(t))
 def save(self):
  r=dict(schema='attention-core-native-v1',complete=len(self.samples)==2 and len(self.calls)==128 and self.counts==Counter({(n,p):c for n in self.modules for p,c in [('prefill',1),('decode',15)]}),modules=self.modules,calls=self.calls,samples=self.samples,counts=[dict(module=n,phase=p,count=c) for (n,p),c in self.counts.items()],artifact_logical_bytes=self.bytes,scope='Two representative valid-prefix banks; no whole KV owner clone or untouched cache proof')
  (self.out/'MANIFEST.json').write_text(json.dumps(r,indent=2)+'\n');return r
 def install(self,model):
  found=[(n,m) for n,m in model.named_modules() if n.endswith('.self_attn.attn')];assert len(found)==8 and len({id(m.impl) for n,m in found})==8
  mods={}
  for name,owner in found:
   impl=owner.impl;orig=impl.forward;mod=sys.modules[orig.__module__];assert type(impl).__name__=='FlashAttentionImpl' and hasattr(mod,'flash_attn_varlen_func')
   mods[id(mod)]=mod;self.modules[name]=dict(implementation=type(impl).__module__+'.'+type(impl).__name__,forward=identity(orig),attrs={k:getattr(impl,k,None) for k in ('head_size','num_heads','num_kv_heads','scale','vllm_flash_attn_version','fa4_hd256','sliding_window','logits_soft_cap','dcp_world_size')})
   def forward(*args,_orig=orig,_name=name,**kwargs):
    assert self.pending is None
    b=inspect.signature(_orig).bind(*args,**kwargs);b.apply_defaults();v=b.arguments;md=v['attn_metadata'];assert md is not None
    starts=md.query_start_loc.detach().cpu().tolist();lens=md.seq_lens.detach().cpu().tolist();phase=phase_from_lists(starts,lens)
    ctx=dict(name=_name,phase=phase,calls=0,owner_output=v['output'],starts=starts,lens=lens);self.pending=ctx
    try:
     result=_orig(*args,**kwargs);assert ctx['calls']==1 and result is v['output'];self.counts[(_name,phase)]+=1;self.save();return result
    finally:self.pending=None
   had='forward' in impl.__dict__;prior=impl.__dict__.get('forward');self.bindings.append((impl,'forward',had,prior,forward));impl.forward=forward
  for mod in mods.values():
   orig=mod.flash_attn_varlen_func;sig=inspect.signature(orig)
   @functools.wraps(orig)
   def helper(*args,_orig=orig,_sig=sig,**kwargs):
    ctx=self.pending
    if ctx is None:return _orig(*args,**kwargs)
    ctx['calls']+=1;assert ctx['calls']==1
    b=_sig.bind(*args,**kwargs);b.apply_defaults();v=b.arguments
    assert v['out'] is not None and v['block_table'] is not None and v['seqused_k'] is not None
    import torch
    q,k,val,out=v['q'],v['k'],v['v'],v['out'];starts=v['cu_seqlens_q'].detach().cpu().tolist();lens=v['seqused_k'].detach().cpu().tolist()
    assert starts==ctx['starts'] and lens==ctx['lens'] and q.shape[0]==starts[-1]
    scalar={};small={}
    for key,x in v.items():
     if key in ('q','k','v','out'):continue
     if isinstance(x,torch.Tensor):small[key]=x
     elif x is None or isinstance(x,(bool,int,float,str,list,tuple)):scalar[key]=x
     else:scalar[key]=dict(callable_identity=identity(x) if callable(x) else str(type(x)))
    desc=dict(module=ctx['name'],phase=ctx['phase'],helper=identity(_orig),scalar_args=scalar,tensors={n:metadata(t) for n,t in dict(q=q,k=k,v=val,out=out,**small).items()},original_calls=1)
    chosen=ctx['name'].endswith('layers.3.self_attn.attn') and ctx['phase'] not in [s['phase'] for s in self.samples]
    if chosen:
     assert q.dtype==k.dtype==val.dtype==torch.bfloat16 and tuple(q.shape[1:])==(16,256) and k.shape==val.shape and tuple(k.shape[2:])==(4,256)
     table=v['block_table'];assert len(lens)==32 and max(lens)<=68 and min(lens)>0
     valid=[]
     for seq,L in enumerate(lens):
      pos=torch.arange(L,device=q.device);blocks=table[seq,pos//k.shape[1]].long();assert bool(((blocks>=0)&(blocks<k.shape[0])).all());valid.append((blocks,pos%k.shape[1]))
     gather=lambda t:torch.cat([t[blocks,offs] for blocks,offs in valid],dim=0)
     kvk,kvv=gather(k),gather(val);tag=ctx['phase'];row=dict(desc,pre_aliases=aliases(dict(q=q,k=k,v=val,out=out)),q=self.tensor(q,tag+'_q'),valid_k=self.tensor(kvk,tag+'_valid_k'),valid_v=self.tensor(kvv,tag+'_valid_v'),metadata={})
     for key,t in small.items():
      assert t.numel()*t.element_size()<=4*1024*1024;row['metadata'][key]=self.tensor(t,tag+'_'+key)
     hashes={key:self.digest(t) for key,t in small.items()};qh=self.digest(q);kh=self.digest(kvk);vh=self.digest(kvv)
    result=_orig(*args,**kwargs)
    if chosen:
     returned=result[0] if isinstance(result,tuple) else result
     assert isinstance(returned,torch.Tensor)
     row['output']=self.tensor(out,tag+'_output');row['helper_return_type']=str(type(result));row['helper_return_out_identity']=returned is out;row['helper_return_identity_preserved']=True;row['returned_output_descriptor']=metadata(returned);row['returned_aliases']=aliases(dict(out=out,returned=returned))
     if isinstance(result,tuple):
      assert len(result)==2;row['lse']=self.tensor(result[1],tag+'_lse')
     row['inputs_preserved']=self.digest(q)==qh and self.digest(gather(k))==kh and self.digest(gather(val))==vh and all(self.digest(small[key])==h for key,h in hashes.items())
     assert row['inputs_preserved'];row['finite_output']=bool(torch.isfinite(out).all());self.samples.append(row)
    self.calls.append(desc);self.save();return result
   self.bindings.append((mod,'flash_attn_varlen_func',True,orig,helper));mod.flash_attn_varlen_func=helper
  return self.save()
 def close(self):
  errors=[]
  for obj,key,had,prior,new in reversed(self.bindings):
   if getattr(obj,key) is not new:errors.append(key);continue
   if had:setattr(obj,key,prior)
   else:delattr(obj,key)
  self.bindings=[];assert not errors
 def verify(self):assert self.save()['complete']
def install(model,out):
 assert not hasattr(model,ATTR);o=Observer(out);setattr(model,ATTR,o)
 try:return o.install(model)
 except BaseException:o.close();delattr(model,ATTR);raise
def finish(model):
 o=getattr(model,ATTR,None)
 if o is None:return {'observer_absent':True}
 try:o.close();o.verify();r=o.save();r['observers_removed']=True;return r
 finally:delattr(model,ATTR)
def close_only(model):
 o=getattr(model,ATTR,None)
 if o is None:return {'observer_absent':True}
 try:o.close();return {'observers_removed':True}
 finally:delattr(model,ATTR)
