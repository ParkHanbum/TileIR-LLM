"""Scoped public producer/consumer observation; preserves original returns."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from gated_norm_runtime_capture_v2.observer import Observer as Base
from norm_runtime_contract_capture_v1.observer import identity,aliases
ATTR='_tileir_attention_gate_observer_v1'
class Observer(Base):
 def save(self):
  r=dict(schema='vllm-attention-sigmoid-gate-v1',complete=len(self.samples)==2 and not self.pending,samples=list(self.samples.values()),weights={},modules=self.modules,observed_calls=self.seen,source_pins=json.loads(Path(__file__).with_name('PINS.json').read_text()),scope='Eager public helper/attention-output/o_proj-input association; no optimized fusion claim')
  (self.out/'MANIFEST.json').write_text(json.dumps(r,indent=2)+'\n');return r
 def install(self,model):
  found=[(n,m) for n,m in model.named_modules() if n.endswith('layers.3.self_attn')];assert len(found)==1
  name,m=found[0];assert m.attn_output_gate is True
  original=m._project_qkv_gate;had='_project_qkv_gate' in m.__dict__;prior=m.__dict__.get('_project_qkv_gate')
  self.modules['gate']=dict(path=name,forward=identity(m.forward),project=identity(original),attention=identity(m.attn.forward),consumer=identity(m.o_proj.forward))
  self.binding=(m,had,prior);self.wrapper=None
  def project(*args,**kwargs):
   assert not self.pending,'Producer overlap';ret=original(*args,**kwargs)
   assert isinstance(ret,tuple) and len(ret)==4;gate=ret[3];assert gate is not None
   shape=tuple(gate.shape);assert shape in ((1696,4096),(32,4096)),('Unsupported actual gate shape',shape)
   phase='prefill' if shape[0]==1696 else 'decode';step=sum(self.seen.values());self.seen[str(shape)]=self.seen.get(str(shape),0)+1
   state=dict(step=step,phase=phase,gate=gate,stage='project',capture=phase not in self.samples)
   if state['capture']:state['gate_record']=self.tensor(gate,phase+'_pre_gate')
   self.pending['call']=state
   return ret
  self.wrapper=project;m._project_qkv_gate=project
  def attn_post(mod,args,kwargs,result):
   assert 'call' in self.pending;state=self.pending['call'];assert state['stage']=='project' and result is not None and tuple(result.shape)==tuple(state['gate'].shape)
   state.update(stage='attention',x=result)
   if state['capture']:
    state['x_record']=self.tensor(result,state['phase']+'_pre_x');state['pre_aliases']=aliases(dict(x=result,gate=state['gate']))
  def consumer_pre(mod,args,kwargs):
   assert 'call' in self.pending;state=self.pending.pop('call');assert state['stage']=='attention'
   y=args[0] if args else kwargs['input'];assert tuple(y.shape)==tuple(state['x'].shape)
   if state['capture']:
    phase=state['phase'];values=dict(x=state['x'],gate=state['gate']);pre=dict(x=state['x_record'],gate=state['gate_record']);post={k:self.tensor(t,phase+'_post_'+k) for k,t in values.items()}
    row=dict(kind='attention_gate',phase=phase,step=state['step'],association=['project_return_gate','attention_return','o_proj_input'],pre=pre,post_inputs=post,input_mutated={k:pre[k]['logical_sha256']!=post[k]['logical_sha256'] for k in pre},pre_aliases=state['pre_aliases'],post_aliases=aliases(dict(values,output=y)),outputs=[self.tensor(y,phase+'_Y')],return_kind='tensor',source_boundary=self.modules['gate'])
    self.samples[phase]=row;self.save()
  try:
   self.handles.extend((m.attn.register_forward_hook(attn_post,with_kwargs=True,always_call=True),m.o_proj.register_forward_pre_hook(consumer_pre,with_kwargs=True)))
   return self.save()
  except BaseException:self.close();raise
 def close(self):
  try:
   if getattr(self,'wrapper',None) is not None:
    m,had,prior=self.binding;assert m.__dict__.get('_project_qkv_gate') is self.wrapper,'Foreign helper binding; not overwritten'
    if had:m._project_qkv_gate=prior
    else:delattr(m,'_project_qkv_gate')
    self.wrapper=None
  finally:super().close()
def install(model,out):
 assert not hasattr(model,ATTR);o=Observer(out);setattr(model,ATTR,o)
 try:return o.install(model)
 except BaseException:o.close();delattr(model,ATTR);raise
def finish(model):
 o=getattr(model,ATTR,None)
 if o is None:return dict(observer_absent=True)
 try:o.close();r=o.save();r['observers_removed']=True;return r
 finally:delattr(model,ATTR)
