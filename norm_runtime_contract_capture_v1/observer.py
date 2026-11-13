"""Public GemmaRMSNorm observers. Never replace inputs or returned objects."""
from pathlib import Path
import hashlib,json
TARGETS={'no_residual':'layers.0.input_layernorm','with_residual':'layers.0.post_attention_layernorm'}
ATTR='_tileirblas_norm_observer_v1'
def identity(fn):
 code=getattr(getattr(fn,'__func__',fn),'__code__',None)
 return dict(module=getattr(fn,'__module__',None),qualname=getattr(fn,'__qualname__',None),source=None if code is None else code.co_filename,line=None if code is None else code.co_firstlineno,source_sha256=None if code is None else hashlib.sha256(Path(code.co_filename).read_bytes()).hexdigest())
def metadata(t):
 import torch
 return dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),dtype=str(t.dtype),device=str(t.device),storage_nbytes=t.untyped_storage().nbytes(),data_ptr=t.data_ptr(),storage_ptr=t.untyped_storage().data_ptr(),object_identity=id(t),stream=int(torch.cuda.current_stream(t.device).cuda_stream))
def aliases(values):
 out={}
 for a,x in values.items():
  for b,y in values.items():
   if a>=b:continue
   xa=x.untyped_storage();ya=y.untyped_storage()
   out[a+'|'+b]=dict(same_object=x is y,same_storage=xa.data_ptr()==ya.data_ptr(),storage_overlap=max(xa.data_ptr(),ya.data_ptr())<min(xa.data_ptr()+xa.nbytes(),ya.data_ptr()+ya.nbytes()))
 return out
class Observer:
 def __init__(self,out):
  self.out=Path(out);self.out.mkdir(exist_ok=False);self.handles=[];self.pending={};self.samples={};self.weights={};self.modules={};self.live={};self.seen={}
 def digest(self,t):
  import torch
  return hashlib.sha256(t.detach().to('cpu').contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 def tensor(self,t,label):
  import torch
  assert isinstance(t,torch.Tensor) and t.dtype in (torch.bfloat16,torch.float32) and t.device.type=='cuda'
  meta=metadata(t);c=t.detach().to('cpu').contiguous();p=self.out/(label+'.pt');assert not p.exists();torch.save(c,p)
  return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),logical_sha256=hashlib.sha256(c.view(torch.uint8).numpy().tobytes()).hexdigest(),metadata=meta,saved_shape=list(c.shape),saved_stride=list(c.stride()))
 def save(self):
  r=dict(schema='vllm-gemma-rms-boundary-v1',complete=len(self.samples)==4 and not self.pending,weights=self.weights,samples=list(self.samples.values()),modules=self.modules,observed_calls=self.seen,source_pins=json.loads(Path(__file__).with_name('PINS.json').read_text()),scope='Eager module facade, not _C in-place ABI; no performance claim')
  (self.out/'MANIFEST.json').write_text(json.dumps(r,indent=2)+'\n');return r
 def install(self,model):
  found={}
  for name,m in model.named_modules():
   for k,suffix in TARGETS.items():
    if name==suffix or name.endswith('.'+suffix):assert k not in found;found[k]=(name,m)
  assert set(found)==set(TARGETS),found
  try:
   for kind,(name,m) in found.items():
    assert type(m).__name__=='GemmaRMSNorm' and tuple(m.weight.shape)==(2560,)
    self.live[kind]=m;self.modules[kind]=dict(path=name,class_name=type(m).__module__+'.'+type(m).__name__,epsilon=m.variance_epsilon,forward=identity(m.forward),selected_forward=identity(getattr(m,'_forward_method',m.forward)),forward_cuda=identity(m.forward_cuda),forward_native=identity(m.forward_native),weight_identity=id(m.weight))
    def pre(mod,args,kwargs,_kind=kind,_name=name):
     x=args[0] if args else kwargs['x'];res=args[1] if len(args)>1 else kwargs.get('residual');shape=tuple(x.shape)
     self.seen[str((_kind,shape,res is not None))]=self.seen.get(str((_kind,shape,res is not None)),0)+1
     if shape not in ((1696,2560),(32,2560)):return
     phase='prefill' if shape[0]==1696 else 'decode';key=_kind+'_'+phase
     if key in self.samples:return
     assert (res is None)==(_kind=='no_residual');assert _kind not in self.pending
     if _kind not in self.weights:self.weights[_kind]=self.tensor(mod.weight,_kind+'_W')
     values=dict(x=x,weight=mod.weight)
     if res is not None:values['residual']=res
     row=dict(kind=_kind,phase=phase,module_path=_name,weight_key=_kind,epsilon=mod.variance_epsilon,pre={k:self.tensor(t,key+'_pre_'+k) for k,t in values.items() if k!='weight'},pre_aliases=aliases(values),selected_forward=identity(getattr(mod,'_forward_method',mod.forward)))
     self.pending[_kind]=(key,row,values);self.save()
    def post(mod,args,kwargs,result,_kind=kind):
     if _kind not in self.pending:return
     key,row,values=self.pending.pop(_kind)
     outputs=list(result) if isinstance(result,tuple) else [result]
     assert len(outputs)==(1 if _kind=='no_residual' else 2)
     row['return_kind']='tuple' if isinstance(result,tuple) else 'tensor';row['output_order']=['normalized'] if len(outputs)==1 else ['normalized','residual_sum']
     row['post_inputs']={k:self.tensor(t,key+'_post_'+k) for k,t in values.items() if k!='weight'}
     row['input_mutated']={k:row['pre'][k]['logical_sha256']!=row['post_inputs'][k]['logical_sha256'] for k in row['pre']}
     row['weight_post_logical_sha256']=self.digest(mod.weight);assert row['weight_post_logical_sha256']==self.weights[_kind]['logical_sha256']
     row['outputs']=[self.tensor(t,key+'_out'+str(i)) for i,t in enumerate(outputs)]
     values.update({'output'+str(i):t for i,t in enumerate(outputs)});row['post_aliases']=aliases(values)
     assert id(mod.weight)==self.modules[_kind]['weight_identity'];self.samples[key]=row;self.save()
    self.handles.extend((m.register_forward_pre_hook(pre,with_kwargs=True),m.register_forward_hook(post,with_kwargs=True)))
   return self.save()
  except BaseException:self.close();raise
 def close(self):
  for h in self.handles:h.remove()
  self.handles=[]
 def verify(self):
  for k,m in self.live.items():
   if k not in self.weights:continue
   d=self.digest(m.weight);self.weights[k]['final_logical_sha256']=d;assert d==self.weights[k]['logical_sha256']
def install(model,out):
 assert not hasattr(model,ATTR);o=Observer(out);setattr(model,ATTR,o)
 try:return o.install(model)
 except BaseException:o.close();delattr(model,ATTR);raise
def finish(model):
 o=getattr(model,ATTR,None)
 if o is None:return dict(observer_absent=True)
 try:o.close();o.verify();r=o.save();r['observers_removed']=True;return r
 finally:delattr(model,ATTR)
