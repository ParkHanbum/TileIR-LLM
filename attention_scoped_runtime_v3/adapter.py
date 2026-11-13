"""Independent native baseline. 96 scoped candidate calls; 32 explicit native exceptions.
Hashes cover Q, metadata and valid KV prefixes, NOT the whole live cache owner.
"""
from pathlib import Path
import inspect,json,functools
from collections import Counter
from attention_core_runtime_capture_v1.observer import Observer,metadata,identity,aliases
ATTR='_attention_scoped_v1'
MODULES={f'language_model.model.layers.{i}.self_attn.attn' for i in (3,7,11,15,19,23,27,31)}
def classify(starts,lens):
 assert len(starts)==33 and starts[0]==0 and len(lens)==32 and len(set(lens))==1
 width=starts[1];assert starts==[i*width for i in range(33)]
 L=lens[0];assert (width==53 and L==53) or (width==1 and 54<=L<=68)
 return L,L<=64

def validate_scalars(v):
 expected=dict(dropout_p=0.0,softmax_scale=.0625,causal=True,softcap=0,deterministic=False,return_attn_probs=False,return_softmax_lse=False,num_splits=0,fa_version=2,cp_world_size=1,cp_rank=0)
 for key,value in expected.items():assert v.get(key)==value,(key,v.get(key))
 assert list(v['window_size'])==[-1,-1]
 for key in ('cu_seqlens_k','q_v','alibi_slopes','scheduler_metadata','output_scale','s_aux','cp_tot_seqused_k','mask_mod','block_sparse_tensors','aux_tensors','aux_tensor_leading_dims','dynamic_causal'):assert v.get(key) is None,key

def census(rows):
 assert len(rows)==128 and all(r['passed'] for r in rows)
 counts=Counter((r['module'],r['length']) for r in rows)
 assert counts==Counter({(m,L):1 for m in MODULES for L in range(53,69)})
 assert sum(r['supported'] for r in rows)==96
 return True

class Adapter(Observer):
 def __init__(self,output):
  super().__init__(output);self.mode=None;self.rows=[];self.failed=False;self.qualified=False
 def save(self):
  r=dict(mode=self.mode,failed=self.failed,rows=self.rows,late_banks=self.samples,artifact_logical_bytes=self.bytes,scope='Q/metadata/valid KV prefixes only; no untouched live owner proof')
  (self.out/'PROGRESS.json').write_text(json.dumps(r,indent=2)+'\n');return r
 def latch(self,fn):
  @functools.wraps(fn)
  def guarded(*args,**kwargs):
   assert not self.failed, 'Latched attention failure; queued reentry forbidden'
   try:return fn(*args,**kwargs)
   except BaseException as e:
    self.failed=True;self.save_error(e);raise
  return guarded
 def install(self,model):
  found=[(n,m) for n,m in model.named_modules() if n.endswith('.self_attn.attn')];assert {n for n,_ in found}==MODULES
  mods={}
  for name,owner in found:
   impl=owner.impl;orig=impl.forward;assert type(impl).__name__=='FlashAttentionImpl';mod=__import__(orig.__module__,fromlist=['x']);mods[id(mod)]=mod
   self.modules[name]=dict(forward=identity(orig),implementation=type(impl).__module__+'.'+type(impl).__name__)
   def forward(*args,_orig=orig,_name=name,**kw):
    assert not self.failed and self.mode in ('shadow','replace') and self.pending is None
    b=inspect.signature(_orig).bind(*args,**kw);b.apply_defaults();md=b.arguments['attn_metadata'];assert md is not None
    starts=md.query_start_loc.detach().cpu().tolist();lens=md.seq_lens.detach().cpu().tolist();L,supported=classify(starts,lens)
    assert not any(r['module']==_name and r['length']==L for r in self.rows)
    ctx=dict(name=_name,length=L,supported=supported,starts=starts,lens=lens,calls=0,out=b.arguments['output']);self.pending=ctx
    try:
     result=_orig(*args,**kw);assert ctx['calls']==1 and result is ctx['out'];return result
    except BaseException as e:self.failed=True;self.save_error(e);raise
    finally:self.pending=None
   forward=self.latch(forward)
   had='forward' in impl.__dict__;prior=impl.__dict__.get('forward');self.bindings.append((impl,'forward',had,prior,forward));impl.forward=forward
  for mod in mods.values():
   orig=mod.flash_attn_varlen_func;sig=inspect.signature(orig)
   @functools.wraps(orig)
   def helper(*args,_orig=orig,_sig=sig,**kw):
    assert self.pending is not None,'Unexpected unowned attention dispatch'
    ctx=self.pending;ctx['calls']+=1;assert ctx['calls']==1 and not self.failed
    b=_sig.bind(*args,**kw);b.apply_defaults()
    try:return self.dispatch(b.arguments,_orig,args,kw,ctx)
    except BaseException as e:self.failed=True;self.save_error(e);raise
   helper=self.latch(helper)
   self.bindings.append((mod,'flash_attn_varlen_func',True,orig,helper));mod.flash_attn_varlen_func=helper
  return dict(modules=self.modules,supported=96,explicit_native_exceptions=32)
 def save_error(self,e):
  self.save();(self.out/'FAILURE.json').write_text(json.dumps(dict(error=repr(e),current=self.pending),default=str,indent=2)+'\n')
 def dispatch(self,v,orig,args,kw,ctx):
  import torch
  from tileir_paged_attention_v8.api import attention
  validate_scalars(v)
  q,k,val,out=v['q'],v['k'],v['v'],v['out'];cuq,lens,table=v['cu_seqlens_q'],v['seqused_k'],v['block_table']
  assert out is not None and out.data_ptr()==ctx['out'].data_ptr()
  assert cuq.cpu().tolist()==ctx['starts'] and lens.cpu().tolist()==ctx['lens']
  assert v['max_seqlen_q']==ctx['starts'][1] and v['max_seqlen_k']==ctx['length']
  assert q.dtype==k.dtype==val.dtype==out.dtype==torch.bfloat16 and q.shape==out.shape and tuple(q.shape[1:])==(16,256)
  assert k.shape==val.shape and tuple(k.shape[1:])==(528,4,256) and k.stride()==val.stride()==(1081344,2048,512,1)
  assert q.stride()==out.stride()==(4096,256,1) and q.storage_offset()==out.storage_offset()==0
  assert k.untyped_storage().data_ptr()==val.untyped_storage().data_ptr() and val.storage_offset()==k.storage_offset()+256
  inputs={n:t for n,t in v.items() if isinstance(t,torch.Tensor) and n!='out'}
  for n,t in inputs.items():
   assert t.device==q.device and not t.requires_grad
   s=t.untyped_storage();o=out.untyped_storage();assert not(o.data_ptr()<s.data_ptr()+s.nbytes() and s.data_ptr()<o.data_ptr()+o.nbytes()),n
  for n in ('q_descale','k_descale','v_descale'):
   t=v[n];assert t.dtype==torch.float32 and t.shape==(32,4) and t.stride()==(0,0) and bool((t==1).all())
  assert table.dtype==cuq.dtype==lens.dtype==torch.int32 and table.shape==(32,8) and table.stride()==(8,1) and cuq.stride()==lens.stride()==(1,)
  valid=[]
  for seq,L in enumerate(ctx['lens']):
   pos=torch.arange(L,device=q.device);blocks=table[seq,pos//528].long();assert bool(((blocks>=0)&(blocks<k.shape[0])).all());valid.append((blocks,pos%528))
  def gather(t):return torch.cat([t[b,p] for b,p in valid],dim=0)
  small={n:t for n,t in inputs.items() if n not in ('q','k','v')}
  assert q.numel()*2+sum(t.numel()*t.element_size() for t in small.values())+sum(ctx['lens'])*4*256*4<=128*1024*1024
  def hashes():return dict(q=self.digest(q),k=self.digest(gather(k)),v=self.digest(gather(val)),**{n:self.digest(t) for n,t in small.items()})
  before=hashes();late=self.mode=='shadow' and ctx['name'].endswith('layers.3.self_attn.attn') and ctx['length'] in (65,68)
  bank=None
  if late:
   tag='length'+str(ctx['length']);bank=dict(module=ctx['name'],length=ctx['length'],helper=identity(orig),scalar_args={n:x for n,x in v.items() if n not in inputs and n!='out'},descriptors={n:metadata(t) for n,t in dict(inputs,out=out).items()},q=self.tensor(q,tag+'_q'),valid_k=self.tensor(gather(k),tag+'_k'),valid_v=self.tensor(gather(val),tag+'_v'),metadata={n:self.tensor(t,tag+'_'+n) for n,t in small.items()})
  native=0;tile=0
  if self.mode=='shadow' or not ctx['supported']:
   result=orig(*args,**kw);native=1;assert result is out
  if ctx['supported']:
   target=torch.empty_like(out) if self.mode=='shadow' else out
   # The native output is never given to the candidate in shadow mode.
   native_hash=self.digest(out) if self.mode=='shadow' else None
   result_tile=attention(q,k,val,cuq,lens,table,target,scale=.0625,causal=True);tile=1;assert result_tile is target
   assert bool(torch.isfinite(target).all())
   if self.mode=='shadow':
    raw=torch.equal(out.view(torch.uint8),target.view(torch.uint8));d=(out.float()-target.float()).abs();strict=bool((d<=1e-5+1e-5*out.float().abs()).all())
    if not raw or not strict:
     for n,t in dict(q=q,native=out,tile=target).items():self.tensor(t,'first_mismatch_'+n)
     assert False,'First supported attention raw/strict mismatch; no replacement'
    assert self.digest(out)==native_hash,'Candidate modified native output'
   else:result=out
  assert bool(torch.isfinite(out).all()) and hashes()==before
  if bank is not None:
   bank.update(output=self.tensor(out,tag+'_out'),inputs_preserved=True,original_return_identity=True);self.samples.append(bank)
  self.rows.append(dict(module=ctx['name'],length=ctx['length'],supported=ctx['supported'],native_calls=native,tile_calls=tile,passed=True,caller_out=metadata(out),input_descriptors={n:metadata(t) for n,t in inputs.items()},coverage='valid-prefix hashes, not whole owner'))
  self.save();return result
 def begin(self,mode):
  assert not self.failed and mode in ('shadow','replace')
  if mode=='replace':assert self.qualified
  else:assert self.mode is None
  self.mode=mode;self.rows=[];return self.save()
 def finish(self):
  assert not self.failed;census(self.rows);assert len(self.samples)==2 and {s['length'] for s in self.samples}=={65,68}
  native=sum(r['native_calls'] for r in self.rows);tile=sum(r['tile_calls'] for r in self.rows);assert tile==96 and native==(128 if self.mode=='shadow' else 32)
  self.qualified=True;r=self.save();r.update(shadow_pass=True,supported_calls=96,tileir_gate_calls=96,native_gate_calls=native,explicit_native_exceptions=32,supported_vendor_fallback_calls=0,total_calls=128);return r

def install(model,output):
 assert not hasattr(model,ATTR);a=Adapter(output);setattr(model,ATTR,a)
 try:return a.install(model)
 except BaseException:a.close();delattr(model,ATTR);raise
def start(model,mode):return getattr(model,ATTR).begin(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 try:a.close();return dict(restored=True)
 finally:delattr(model,ATTR)
