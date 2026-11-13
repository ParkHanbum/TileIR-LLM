"""Native-only public KV insertion observation; no backend selection changes."""
from pathlib import Path
import sys,json,hashlib,inspect,functools
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from norm_runtime_contract_capture_v1.observer import identity,metadata,aliases
ATTR='_attention_kv_capture_v1';CAP=256*1024*1024;ITEM_CAP=64*1024*1024

def phase_from_lists(starts,lengths):
 q=[b-a for a,b in zip(starts,starts[1:])]
 assert len(q)==len(lengths)==32 and starts[0]==0
 if q==[53]*32 and lengths==[53]*32:return 'prefill'
 if q==[1]*32 and len(set(lengths))==1 and 54<=lengths[0]<=68:return 'decode'
 raise AssertionError('Unsupported actual query/sequence lengths')
def valid_slots(slots,capacity):
 assert all(type(x) is int and 0<=x<capacity for x in slots),'Negative/out-of-range slot unsupported in first capture'
 assert len(slots)==len(set(slots)),'Duplicate slot unsupported'
 return slots
class Observer:
 def __init__(self,out):
  self.out=Path(out);self.out.mkdir(exist_ok=False);self.modules={};self.counts=Counter();self.samples=[];self.bindings=[];self.pending=None;self.closed=False;self.bytes=0
 def snapshot(self,t,label):
  import torch
  n=t.numel()*t.element_size();assert n<=ITEM_CAP and self.bytes+n<=CAP
  cpu=t.detach().cpu().contiguous();self.bytes+=n
  p=self.out/(label+'.pt');assert not p.exists();torch.save(cpu,p)
  return dict(metadata=metadata(t),path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),logical_sha256=hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
 def progress(self):
  rows=[dict(module=n,phase=p,count=c) for (n,p),c in self.counts.items()]
  done=len(self.modules)==8 and self.counts==Counter({(n,p):c for n in self.modules for p,c in [('prefill',1),('decode',15)]}) and len(self.samples)==2 and self.pending is None
  r=dict(schema='attention-kv-capture-v1',complete=done,modules=self.modules,counts=rows,samples=self.samples,observers_removed=self.closed,artifact_logical_bytes=self.bytes,scope='Selected addressed slots only; no untouched shared backing preservation proof')
  (self.out/'PROGRESS.json').write_text(json.dumps(r,indent=2));return r
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
    selected=ctx['name'].endswith('layers.3.self_attn.attn') and not any(s['phase']==ctx['phase'] for s in self.samples)
    if not selected:return _original(*args,**kwargs)
    return self.capture(ctx,v,lambda:_original(*args,**kwargs),_original)
   self.bindings.append((mod,'reshape_and_cache_flash',True,original,helper));mod.reshape_and_cache_flash=helper
  return self.progress()
 def capture(self,ctx,v,native,original):
  import torch
  key,value,kc,vc,slots=[v[k] for k in ('key','value','key_cache','value_cache','slot_mapping')]
  assert v['kv_cache_dtype'] in ('auto','bfloat16') and key.dtype==value.dtype==kc.dtype==vc.dtype==torch.bfloat16
  assert kc.ndim==vc.ndim==4 and kc.shape==vc.shape and tuple(kc.shape[2:])==(4,256)
  values=valid_slots(slots.detach().cpu().tolist(),kc.shape[0]*kc.shape[1]);assert len(values) in (1696,32)
  assert key.shape[0]>=len(values) and value.shape[0]>=len(values)
  # Before any selected-cache gather, bound logical size. Never transfer persistent backing.
  assert len(values)*4*256*kc.element_size()<=ITEM_CAP
  idx=slots.to(torch.int64);blocks=idx//kc.shape[1];offsets=idx%kc.shape[1]
  label=ctx['phase'];inputs={k:v[k] for k in ('key','value','slot_mapping','k_scale','v_scale')}
  row=dict(module=ctx['name'],phase=label,phase_evidence=ctx['metadata'],helper=identity(original),kv_cache_dtype=v['kv_cache_dtype'],owner_cache_descriptor=metadata(ctx['args']['kv_cache']),cache_descriptors={'key':metadata(kc),'value':metadata(vc)},aliases=aliases(dict(inputs,key_cache=kc,value_cache=vc)),slots=values)
  (self.out/'PENDING.json').write_text(json.dumps(row,indent=2))
  row['pre_inputs']={k:self.snapshot(t,label+'_pre_'+k) for k,t in inputs.items()}
  row['pre_slots']={k:self.snapshot(t[blocks,offsets],label+'_pre_'+k) for k,t in [('key_cache',kc),('value_cache',vc)]}
  ret=native();assert ret is None
  row['post_inputs']={k:self.snapshot(t,label+'_post_'+k) for k,t in inputs.items()}
  row['inputs_preserved']={k:row['pre_inputs'][k]['logical_sha256']==row['post_inputs'][k]['logical_sha256'] for k in inputs}
  assert all(row['inputs_preserved'].values())
  row['post_slots']={k:self.snapshot(t[blocks,offsets],label+'_post_'+k) for k,t in [('key_cache',kc),('value_cache',vc)]}
  row.update(original_calls=1,return_is_none=True,return_identity_preserved=True,untouched_cache_verified=False)
  self.samples.append(row);return ret
 def close(self):
  errors=[]
  for obj,name,had,old,new in reversed(self.bindings):
   if getattr(obj,name) is not new:errors.append(name);continue
   if had:setattr(obj,name,old)
   else:delattr(obj,name)
  self.bindings=[];self.closed=not errors;assert not errors,errors
 def finish(self):
  self.close();r=self.progress();assert r['complete'];p=self.out/'MANIFEST.json'
  with p.open('x') as f:json.dump(r,f,indent=2)
  return r
def install(model,out):
 assert not hasattr(model,ATTR);o=Observer(out);setattr(model,ATTR,o)
 try:return o.install(model)
 except BaseException:o.close();delattr(model,ATTR);raise
def finish(model):
 o=getattr(model,ATTR,None)
 if o is None:return {'observer_absent':True}
 try:return o.finish()
 finally:delattr(model,ATTR)
def close_only(model):
 o=getattr(model,ATTR,None)
 if o is None:return {'observer_absent':True}
 try:o.close();return {'observers_removed':True}
 finally:delattr(model,ATTR)
