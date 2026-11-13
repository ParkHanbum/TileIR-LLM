"""Observe selected public ChunkGatedDeltaRule boundary; no dispatch substitution."""
from pathlib import Path
import sys,json,hashlib,inspect,functools
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from norm_runtime_contract_capture_v1.observer import identity,metadata,aliases
from gdn_post_conv_runtime_capture_v1.observer import phase_meta
ATTR='_gdn_chunk_capture_v1';CAP=512*1024*1024;SINGLE_STORAGE_CAP=128*1024*1024
class Observer:
 def __init__(self,out):
  self.out=Path(out);self.out.mkdir(exist_ok=False);self.modules={};self.counts={};self.samples=[];self.bindings=[];self.errors=[];self.pending=False;self.bytes=0;self.closed=False
 def tensor(self,t,label):
  import torch
  # Guard BEFORE transfer: never duplicate the persistent shared multi-GiB owner.
  assert t.untyped_storage().nbytes()<=SINGLE_STORAGE_CAP,'Unexpected shared/large backing; descriptors needed before capture'
  n=t.numel()*t.element_size();assert self.bytes+n<=CAP
  cpu=t.detach().cpu().clone();self.bytes+=n;p=self.out/(label+'.pt');assert not p.exists();torch.save(cpu,p)
  return dict(metadata=metadata(t),path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),logical_sha256=hashlib.sha256(cpu.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
 def progress(self):
  r=dict(schema='gdn-chunk-capture-v1',complete=len(self.samples)==1 and len(self.counts)==24 and all(x==1 for x in self.counts.values()) and not self.pending and not self.errors,modules=self.modules,counts=self.counts,samples=self.samples,errors=self.errors,observers_removed=self.closed,artifact_logical_bytes=self.bytes)
  (self.out/'PROGRESS.json').write_text(json.dumps(r,indent=2)+'\n');return r
 def install(self,model):
  from vllm.forward_context import get_forward_context
  named=dict(model.named_modules())
  for name,m in named.items():
   if not name.endswith('.linear_attn.chunk_gated_delta_rule'):continue
   owner=named[name.rsplit('.',1)[0]];original=m._forward_method;sig=inspect.signature(original)
   self.modules[name]=dict(selected=identity(original),backend=str(m.gdn_prefill_backend),owner_prefix=owner.prefix)
   @functools.wraps(original)
   def wrapper(*args,_name=name,_owner=owner,_original=original,_sig=sig,**kwargs):
    import torch
    ctx=get_forward_context();assert isinstance(ctx.attn_metadata,dict)
    phase=phase_meta(ctx.attn_metadata.get(_owner.prefix));assert phase['phase']=='prefill' and not phase['speculative']
    self.counts[_name]=self.counts.get(_name,0)+1;assert self.counts[_name]==1
    b=_sig.bind(*args,**kwargs);b.apply_defaults();v=b.arguments
    if not _name.endswith('layers.0.linear_attn.chunk_gated_delta_rule'):
     result=_original(*args,**kwargs);self.progress();return result
    assert not self.pending and not self.samples;self.pending=True
    tensors={k:t for k,t in v.items() if isinstance(t,torch.Tensor)}
    scalar={k:t for k,t in v.items() if k not in tensors}
    assert all(t is None or type(t) in (bool,int,float,str) for t in scalar.values()),'Uncaptured opaque metadata'
    row=dict(module=_name,phase=phase,helper=identity(_original),arguments=scalar,input_descriptors={k:metadata(t) for k,t in tensors.items()},persistent_views=[metadata(t) for t in _owner.kv_cache],persistent_scope='Descriptors only: no shared owner snapshot; gather and later scatter outside this call',pre_aliases=aliases(tensors))
    # Preserve descriptors before potentially rejecting oversized backing.
    self.progress();(self.out/'PENDING.json').write_text(json.dumps(row,indent=2))
    try:
     row['pre']={k:self.tensor(t,'pre_'+k) for k,t in tensors.items()}
     result=_original(*args,**kwargs) # exactly once; selected provider unchanged
     assert isinstance(result,tuple) and len(result)==2 and all(isinstance(t,torch.Tensor) for t in result)
     row['post_inputs']={k:self.tensor(t,'post_'+k) for k,t in tensors.items()}
     row['input_logical_preserved']={k:row['pre'][k]['logical_sha256']==row['post_inputs'][k]['logical_sha256'] for k in tensors}
     row['outputs']={k:self.tensor(t,'out_'+k) for k,t in zip(('o','final_state'),result)}
     row['aliases']=aliases(dict(tensors,output_o=result[0],output_final_state=result[1]));row.update(original_calls=1,return_identity_preserved=True,return_order=['o','final_state'])
     self.samples.append(row);self.pending=False;self.progress();return result
    except BaseException as e:self.errors.append(repr(e));self.progress();raise
   had='_forward_method' in m.__dict__;prior=m.__dict__.get('_forward_method');self.bindings.append((m,had,prior,wrapper));m._forward_method=wrapper
  assert len(self.modules)==24;return self.progress()
 def close(self):
  conflicts=[]
  for m,had,old,new in reversed(self.bindings):
   if m.__dict__.get('_forward_method') is not new:conflicts.append(type(m).__name__);continue
   if had:m._forward_method=old
   else:delattr(m,'_forward_method')
  self.bindings=[];self.closed=not conflicts;assert not conflicts,conflicts
 def finish(self):
  self.close();r=self.progress();assert r['complete'] and set(self.counts)==set(self.modules)
  with (self.out/'MANIFEST.json').open('x') as f:json.dump(r,f,indent=2)
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
