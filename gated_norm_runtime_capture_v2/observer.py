"""Observe public RMSNormGated facade; original output is never replaced."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from norm_runtime_contract_capture_v1.observer import Observer as Base,identity,aliases
ATTR='_tileirblas_gated_norm_observer_v2'
def original_tensor_descriptor(t):
 return dict(requires_grad=bool(t.requires_grad),is_leaf=bool(t.is_leaf),python_type=type(t).__module__+'.'+type(t).__qualname__)
class Observer(Base):
 def tensor(self,t,label):
  original=original_tensor_descriptor(t)
  record=super().tensor(t,label)
  record['metadata'].update(original)
  return record
 def save(self):
  r=dict(schema='vllm-gated-norm-boundary-v1',complete=len(self.samples)==2 and not self.pending,weights=self.weights,samples=list(self.samples.values()),modules=self.modules,observed_calls=self.seen,source_pins=json.loads(Path(__file__).with_name('PINS.json').read_text()),scope='Public eager module facade; selected forward observed, optimized fusion identity not assumed')
  (self.out/'MANIFEST.json').write_text(json.dumps(r,indent=2)+'\n');return r
 def install(self,model):
  found=[(n,m) for n,m in model.named_modules() if n.endswith('layers.0.linear_attn.norm')]
  assert len(found)==1;name,m=found[0];assert type(m).__name__=='RMSNormGated' and tuple(m.weight.shape)==(128,)
  assert m.bias is None and m.group_size is None and m.norm_before_gate is True and m.activation in ('silu','swish')
  self.live['gated']=m;self.modules['gated']=dict(path=name,epsilon=m.eps,group_size=m.group_size,norm_before_gate=m.norm_before_gate,activation=m.activation,bias=None,selected_forward=identity(getattr(m,'_forward_method',m.forward)),forward_native=identity(m.forward_native),forward_cuda=identity(m.forward_cuda),weight_identity=id(m.weight))
  def pre(mod,args,kwargs):
   x=args[0] if args else kwargs['x'];z=args[1] if len(args)>1 else kwargs.get('z');shape=tuple(x.shape)
   self.seen[str(shape)]=self.seen.get(str(shape),0)+1
   if shape not in ((1696,32,128),(32,32,128)):return
   phase='prefill' if shape[0]==1696 else 'decode'
   if phase in self.samples:return
   assert not self.pending and z is not None and tuple(z.shape)==shape
   if 'gated' not in self.weights:self.weights['gated']=self.tensor(mod.weight,'W')
   values=dict(x=x,z=z,weight=mod.weight)
   row=dict(kind='gated',phase=phase,module_path=name,weight_key='gated',epsilon=mod.eps,pre={k:self.tensor(t,phase+'_pre_'+k) for k,t in values.items() if k!='weight'},pre_aliases=aliases(values),selected_forward=identity(getattr(mod,'_forward_method',mod.forward)))
   self.pending['gated']=(phase,row,values);self.save()
  def post(mod,args,kwargs,result):
   if not self.pending:return
   phase,row,values=self.pending.pop('gated')
   if result is None:self.save();return
   assert not isinstance(result,tuple) and tuple(result.shape)==tuple(values['x'].shape)
   row['post_inputs']={k:self.tensor(t,phase+'_post_'+k) for k,t in values.items() if k!='weight'}
   row['input_mutated']={k:row['pre'][k]['logical_sha256']!=row['post_inputs'][k]['logical_sha256'] for k in row['pre']}
   row['outputs']=[self.tensor(result,phase+'_Y')];row['return_kind']='tensor';row['output_order']=['gated_normalized']
   values['output']=result;row['post_aliases']=aliases(values)
   row['weight_post_logical_sha256']=self.digest(mod.weight);assert row['weight_post_logical_sha256']==self.weights['gated']['logical_sha256']
   assert id(mod.weight)==self.modules['gated']['weight_identity'];self.samples[phase]=row;self.save()
  try:self.handles.extend((m.register_forward_pre_hook(pre,with_kwargs=True),m.register_forward_hook(post,with_kwargs=True,always_call=True)));return self.save()
  except BaseException:self.close();raise
def install(model,out):
 assert not hasattr(model,ATTR);o=Observer(out);setattr(model,ATTR,o)
 try:return o.install(model)
 except BaseException:o.close();delattr(model,ATTR);raise
def finish(model):
 o=getattr(model,ATTR,None)
 if o is None:return dict(observer_absent=True)
 try:o.close();o.verify();r=o.save();r['observers_removed']=True;return r
 finally:delattr(model,ATTR)
