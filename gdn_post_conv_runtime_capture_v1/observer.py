"""Public prefill-only post-conv helper capture; no replacement or GPU at import."""
from pathlib import Path
import sys,json,hashlib,inspect,functools
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from norm_runtime_contract_capture_v1.observer import identity,metadata,aliases
ATTR='_gdn_post_conv_capture_v1'
INPUTS=('conv_output','a','b','A_log','dt_bias')
OUTPUTS=('q','k','v','g','beta')
CAP=512*1024*1024

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def phase_meta(m):
 assert m is not None,'Missing actual forward metadata'
 d={k:int(getattr(m,k)) for k in ('num_prefills','num_decodes','num_actual_tokens')}
 d['speculative']=getattr(m,'spec_sequence_masks') is not None
 d['phase']='prefill' if d['num_prefills']>0 and d['num_decodes']==0 else 'decode' if d['num_prefills']==0 and d['num_decodes']>0 else 'mixed'
 return d

def fullbytes(t):
 import torch
 return torch.empty(0,dtype=torch.uint8,device=t.device).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,)).cpu().clone()
def digest(t):return hashlib.sha256(fullbytes(t).numpy().tobytes()).hexdigest()

class Observer:
 def __init__(self,out):
  self.out=Path(out);self.out.mkdir(exist_ok=False);self.modules={};self.live={};self.counts={};self.params={};self.samples=[];self.bindings=[];self.pending=False;self.errors=[];self.bytes=0;self.closed=False
 def tensor(self,t,label):
  import torch
  meta=metadata(t);raw=fullbytes(t);logical=t.detach().cpu().contiguous().clone()
  n=raw.numel()+logical.numel()*logical.element_size();assert self.bytes+n<=CAP,'Capture size cap'
  self.bytes+=n
  p=self.out/(label+'.pt');r=self.out/(label+'.storage.pt');assert not p.exists() and not r.exists()
  torch.save(logical,p);torch.save(raw,r)
  return dict(metadata=meta,path=str(p),sha256=sha(p),storage_path=str(r),storage_sha256=sha(r),storage_byte_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest())
 def progress(self):
  r=dict(schema='gdn-post-conv-capture-v1',complete=len(self.samples)==1 and len(self.counts)==24 and all(v==1 for v in self.counts.values()) and not self.pending and not self.errors,modules=self.modules,counts=self.counts,parameter_checks=self.params,samples=self.samples,errors=self.errors,observers_removed=self.closed,artifact_bytes=self.bytes)
  (self.out/'PROGRESS.json').write_text(json.dumps(r,indent=2)+'\n');return r
 def hook(self,q,get_context):
  original=q.fused_post_conv_prep;sig=inspect.signature(original)
  @functools.wraps(original)
  def wrapped(*args,**kwargs):
   bound=sig.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments
   matches=[(n,m) for n,m in self.live.items() if v['A_log'] is m.A_log and v['dt_bias'] is m.dt_bias]
   assert len(matches)==1,'Unknown/ambiguous parameter owner'
   name,m=matches[0];ctx=get_context();md=ctx.attn_metadata
   assert isinstance(md,dict);pm=phase_meta(md.get(m.prefix))
   assert pm['phase']=='prefill' and not pm['speculative'],'Unexpected public helper phase recorded as HOLD'
   self.counts[name]=self.counts.get(name,0)+1
   assert self.counts[name]==1,'More than one prefill call per layer'
   before={k:digest(v[k]) for k in ('A_log','dt_bias')}
   self.params[name]={'before':before,'phase':pm}
   selected=name.endswith('layers.0.linear_attn')
   row=None
   if selected:
    assert not self.pending and not self.samples;self.pending=True
    row=dict(module=name,phase=pm,helper=identity(original),arguments={k:v[k] for k in ('num_k_heads','head_k_dim','head_v_dim','apply_l2norm','output_g_exp')},pre={k:self.tensor(v[k],'pre_'+k) for k in INPUTS},pre_aliases=aliases({k:v[k] for k in INPUTS}))
   self.progress()
   try:
    result=original(*args,**kwargs) # exactly once; exact returned object retained
    after={k:digest(v[k]) for k in before};self.params[name]['after']=after;self.params[name]['preserved']=before==after
    assert before==after,'Native parameter mutation'
    if selected:
     assert isinstance(result,tuple) and len(result)==5
     row['post_inputs']={k:self.tensor(v[k],'post_'+k) for k in INPUTS}
     row['input_storage_preserved']={k:row['pre'][k]['storage_byte_sha256']==row['post_inputs'][k]['storage_byte_sha256'] for k in INPUTS}
     assert all(row['input_storage_preserved'].values()),'Native helper input mutation'
     row['outputs']={k:self.tensor(t,'out_'+k) for k,t in zip(OUTPUTS,result)}
     row['aliases']=aliases(dict({k:v[k] for k in INPUTS},**dict(zip(OUTPUTS,result))))
     row.update(original_calls=1,return_kind='tuple',return_order=list(OUTPUTS),return_identity_preserved=True)
     self.samples.append(row);self.pending=False
    self.progress();return result
   except BaseException as e:self.errors.append(repr(e));self.progress();raise
  q.fused_post_conv_prep=wrapped;self.bindings.append((q,'fused_post_conv_prep',original,wrapped))
 def install(self,model):
  import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q
  from vllm.forward_context import get_forward_context
  for name,m in model.named_modules():
   if name.endswith('.linear_attn') and hasattr(m,'A_log') and hasattr(m,'dt_bias'):
    self.live[name]=m;self.modules[name]=dict(prefix=m.prefix,class_name=type(m).__module__+'.'+type(m).__name__,A_log=metadata(m.A_log),dt_bias=metadata(m.dt_bias))
  assert len(self.live)==24
  self.hook(q,get_forward_context);return self.progress()
 def close(self):
  conflicts=[]
  for q,k,old,new in reversed(self.bindings):
   if getattr(q,k) is new:setattr(q,k,old)
   else:conflicts.append(k)
  self.bindings=[];self.closed=not conflicts
  if conflicts:raise RuntimeError('Foreign binding preserved: '+str(conflicts))
 def finish(self):
  self.close();r=self.progress();assert r['complete'],'Incomplete public helper census'
  assert set(self.counts)==set(self.modules) and all(p['preserved'] for p in self.params.values())
  r['source_pins']=json.loads(Path(__file__).with_name('PINS.json').read_text())
  with (self.out/'MANIFEST.json').open('x') as f:json.dump(r,f,indent=2)
  r['manifest_path']=str(self.out/'MANIFEST.json');r['manifest_sha256']=sha(self.out/'MANIFEST.json');return r

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
