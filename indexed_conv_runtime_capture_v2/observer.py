"""Exactly-once public imported helper observer. No replacements or state writes."""
from pathlib import Path
import sys,json,hashlib,inspect,functools
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from norm_runtime_contract_capture_v1.observer import identity,metadata,aliases
ATTR='_indexed_conv_capture_v2'
STATE_CAP=128*1024*1024
FILE_CAP=512*1024*1024

def classify_slots(indices,capacity,null):
 assert isinstance(capacity,int) and capacity>0
 assert all(isinstance(x,int) and not isinstance(x,bool) for x in indices)
 valid=[x for x in indices if x!=null]
 assert all(0<=x<capacity for x in valid),'slot out of bounds'
 assert len(set(valid))==len(valid),'duplicate valid slots outside this capture scope'
 return sorted(valid),[i for i,x in enumerate(indices) if x==null]

def digest(t):
 import torch
 return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def metadata_tree(value,save_tensor,label,is_tensor):
 if is_tensor(value):return {'type':'tensor','value':save_tensor(value,label)}
 if isinstance(value,dict):
  return {'type':'dict','items':[[metadata_tree(k,save_tensor,label+'_key'+str(i),is_tensor),metadata_tree(v,save_tensor,label+'_val'+str(i),is_tensor)] for i,(k,v) in enumerate(value.items())]}
 if isinstance(value,(list,tuple)):
  return {'type':type(value).__name__,'items':[metadata_tree(v,save_tensor,label+'_'+str(i),is_tensor) for i,v in enumerate(value)]}
 assert value is None or type(value) in (str,int,float,bool),'unknown metadata member'
 return {'type':type(value).__name__,'value':value}

def helper_metadata(meta,save_tensor,label,is_tensor):
 if meta is None:return None
 return {k:metadata_tree(getattr(meta,k),save_tensor,label+'_'+k,is_tensor) for k in ('nums_dict','batch_ptr','token_chunk_offset_ptr')}

class Observer:
 def __init__(self,out):
  self.out=Path(out);self.out.mkdir(exist_ok=False);self.samples={};self.counts={};self.bindings=[];self.weights={};self.bytes=0;self.modules={};self.pending=False
 def tensor(self,t,label):
  import torch
  c=t.detach().cpu().contiguous();n=c.numel()*c.element_size()
  assert self.bytes+n<=FILE_CAP,'artifact logical byte budget'
  p=self.out/(label+'.pt');assert not p.exists();torch.save(c,p);self.bytes+=n
  return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),logical_sha256=digest(c),metadata=metadata(t) if t.is_cuda else dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device)),logical_bytes=n)
 def save(self):
  r=dict(schema='indexed-causal-conv-native-v1',complete=len(self.samples)==2 and not self.pending and sum(v for k,v in self.counts.items() if k.endswith(':prefill'))==24 and sum(v for k,v in self.counts.items() if k.endswith(':decode'))==360,modules=self.modules,samples=list(self.samples.values()),counts=self.counts,artifact_logical_bytes=self.bytes,logical_state_cap=STATE_CAP,artifact_cap=FILE_CAP,source_pins=json.loads(Path(__file__).with_name('PINS.json').read_text()))
  (self.out/'MANIFEST.json').write_text(json.dumps(r,indent=2)+'\n');return r
 def install(self,model):
  import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q
  for name,m in model.named_modules():
   if name.endswith('.linear_attn') and hasattr(m,'conv1d'):
    w=m.conv1d.weight;assert tuple(w.shape)==(8192,1,4)
    key=w.data_ptr();assert key not in self.weights
    self.weights[key]=name;self.modules[name]=dict(weight=metadata(w),class_name=type(m).__name__)
  assert len(self.weights)==24
  try:
   for api,phase in [('causal_conv1d_fn','prefill'),('causal_conv1d_update','decode')]:
    original=getattr(q,api);sig=inspect.signature(original)
    @functools.wraps(original)
    def wrapped(*args,_original=original,_sig=sig,_api=api,_phase=phase,**kwargs):
     b=_sig.bind(*args,**kwargs);b.apply_defaults();v=b.arguments
     x,w=v['x'],v['weight'];name=self.weights.get(w.data_ptr());assert name is not None,'unknown weight owner'
     key=name+':'+_phase;self.counts[key]=self.counts.get(key,0)+1
     if not name.endswith('layers.0.linear_attn') or _phase in self.samples:return _original(*args,**kwargs)
     assert not self.pending;self.pending=True
     state=v['conv_states' if _phase=='prefill' else 'conv_state'];idx=v['cache_indices' if _phase=='prefill' else 'conv_state_indices']
     assert state.ndim==3 and state.shape[1]==8192 and state.shape[2]>=3
     assert state.numel()*state.element_size()<=STATE_CAP,'logical state snapshot budget'
     assert idx is not None and idx.ndim==1 and len(idx)==32
     for k in ('num_accepted_tokens','block_idx_last_scheduled_token','initial_state_idx','block_idx_first_scheduled_token','num_computed_tokens'):
      assert v.get(k) is None,('unsupported speculative/APC scope',k)
     assert v.get('out') is None and v.get('block_size_to_align',0)==0
     null=v['null_block_id'];slots,skips=classify_slots(idx.cpu().tolist(),state.shape[0],null)
     # Conservatively reject prefill padding: pad and null processing need separate capture.
     if _phase=='prefill':assert not skips and v.get('pad_slot_id') not in idx.cpu().tolist()
     before=state.detach().cpu().contiguous();untouched=[i for i in range(state.shape[0]) if i not in slots]
     pre_unselected=digest(before[untouched]);values={'x':x,'weight':w,'state':state}
     if v.get('bias') is not None:values['bias']=v['bias']
     row=dict(api=_api,phase=_phase,module=name,helper=identity(_original),arguments={},state_descriptor=metadata(state),slots=slots,skip_positions=skips,out_omitted= 'out' not in kwargs,pre_aliases=aliases(values),pre={},unselected_before_sha=pre_unselected)
     import torch
     for k,t in v.items():
      if isinstance(t,torch.Tensor) and k not in ('conv_states','conv_state'):row['pre'][k]=self.tensor(t,_phase+'_pre_'+k)
      elif not isinstance(t,torch.Tensor):row['arguments'][k]=t if isinstance(t,(str,int,float,bool,type(None))) else {'type':type(t).__module__+'.'+type(t).__name__}
     row['selected_state_before']=self.tensor(before[slots],_phase+'_state_before');del before
     row['metadata_pre']=helper_metadata(v.get('metadata'),self.tensor,_phase+'_metadata_pre',lambda t:isinstance(t,torch.Tensor))
     self.save()
     result=_original(*args,**kwargs) # exactly once, return unchanged
     row['metadata_post']=helper_metadata(v.get('metadata'),self.tensor,_phase+'_metadata_post',lambda t:isinstance(t,torch.Tensor))
     after=state.detach().cpu().contiguous();row['selected_state_after']=self.tensor(after[slots],_phase+'_state_after')
     row['unselected_after_sha']=digest(after[untouched]);del after
     assert row['unselected_after_sha']==pre_unselected,'unselected logical state mutated'
     row['post_x']=self.tensor(x,_phase+'_post_x');row['output']=self.tensor(result,_phase+'_output')
     row['post_aliases']=aliases(dict(values,output=result));row['weight_unchanged']=digest(w)==row['pre']['weight']['logical_sha256'];assert row['weight_unchanged']
     row['state_unselected_unchanged']=True;row['original_calls']=1
     self.samples[_phase]=row;self.pending=False;self.save();return result
    setattr(q,api,wrapped);self.bindings.append((q,api,original,wrapped))
   return self.save()
  except BaseException:self.close();raise
 def close(self):
  for q,k,orig,wrapper in reversed(self.bindings):
   assert getattr(q,k) is wrapper;setattr(q,k,orig)
  self.bindings=[]
 def verify(self):
  assert set(self.counts)=={n+':'+p for n in self.modules for p in ('prefill','decode')}
  assert all(v==(1 if k.endswith(':prefill') else 15) for k,v in self.counts.items())
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
