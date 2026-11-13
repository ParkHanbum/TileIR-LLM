"""Public sampler functions, unchanged calls and return objects. No replacement."""
from pathlib import Path
import sys,json,hashlib,inspect
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from tp1_embedding_runtime_capture_v2.observer import meta,digest
from norm_runtime_contract_capture_v1.observer import identity,aliases
from block_gather_runtime_capture_v1.observer import phase_metadata
ATTR='_greedy_sampler_capture_v1';CAP=256*1024*1024
class Observer:
 def __init__(self,out):
  self.out=Path(out);self.out.mkdir(exist_ok=False);self.bindings=[];self.pending=None;self.rows=[];self.samples=[];self.bytes=0;self.closed=False;self.source={}
 def tensor(self,t,label):
  import torch
  size=t.numel()*t.element_size();assert size<=64*1024*1024 and self.bytes+size<=CAP,'Before-copy artifact bound'
  descriptor=meta(t);c=t.detach().cpu().contiguous().clone();self.bytes+=size;f=self.out/(label+'.pt');assert not f.exists();torch.save(c,f)
  return dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),logical_sha256=digest(c),metadata=descriptor)
 def invoke(self,kind,fn,args,kwargs):
  import torch
  ctx=self.pending;assert ctx is not None;ctx['counts'][kind]+=1;assert ctx['counts'][kind]==1
  bound=inspect.signature(fn).bind(*args,**kwargs);bound.apply_defaults();values=bound.arguments;logits=values['logits'];assert tuple(logits.shape)==(32,248320)
  row=dict(kind=kind,inputs={k:meta(v) for k,v in values.items() if isinstance(v,torch.Tensor)},scalars={k:v for k,v in values.items() if v is None or type(v) in (int,float,bool,str)},original_calls=0)
  if kind=='gumbel':
   assert values['apply_temperature'] is False and values['is_drafting'] is False and values['use_fp64'] is False and values['logits_cache'] is None
   idx=values['expanded_idx_mapping'];temps=values['temperature'].index_select(0,idx.long());assert bool((temps==0).all());row['all_active_temperature_zero']=True
  else:assert values['num_logprobs']==5 and values['max_per_req_token_ids']==0 and values['logits_mode'] is False and values['cu_num_logits'] is None
  selected=ctx['index'] in (0,15);sample=None;before={}
  if selected:
   tensors={k:v for k,v in values.items() if isinstance(v,torch.Tensor)}
   sample=dict(call_index=ctx['index'],phase=ctx['batch']['phase'],kind=kind,inputs={k:self.tensor(v,f"{ctx['index']}_{kind}_in_{k}") for k,v in tensors.items()})
   before={k:digest(v) for k,v in tensors.items()}
   finite=torch.isfinite(logits);row['logits_nonfinite']=dict(nan=int(torch.isnan(logits).sum()),positive_inf=int(torch.isposinf(logits).sum()),negative_inf=int(torch.isneginf(logits).sum()))
   mx=logits.max(dim=-1).values;row['max_tie_counts']=(logits==mx[:,None]).sum(-1).cpu().tolist()
  ret=fn(*args,**kwargs);row['original_calls']=1
  outputs={'sampled':ret} if kind=='gumbel' else {k:getattr(ret,k) for k in ('logprob_token_ids','logprobs','selected_token_ranks')}
  row['outputs']={k:meta(v) for k,v in outputs.items()};row['return_type']=type(ret).__module__+'.'+type(ret).__name__
  if kind=='gumbel':ctx['sampled']=ret
  else:assert values['sampled_token_ids'] is ctx['sampled'],'Same-call sampled association'
  if selected:
   row['input_logical_preserved']={k:digest(values[k])==v for k,v in before.items()}
   assert all(row['input_logical_preserved'].values())
   sample['outputs']={k:self.tensor(v,f"{ctx['index']}_{kind}_out_{k}") for k,v in outputs.items()}
   sample['aliases']=aliases({**tensors,**outputs});sample['observations']=row;self.samples.append(sample)
  ctx['helpers'].append(row);return ret
 def install_classes(self,S,module):
  original_call=S.__call__;self.source={'sampler':identity(original_call),'gumbel':identity(module.gumbel_sample),'topk':identity(module.compute_topk_scores)}
  def call(owner,logits,batch,*a,**kw):
   assert self.pending is None
   ctx=dict(index=len(self.rows),batch=phase_metadata(batch),counts=dict(gumbel=0,topk=0),helpers=[])
   assert ctx['index']<16 and ctx['batch']['num_reqs']==32
   assert ctx['batch']['phase']==('prefill' if ctx['index']==0 else 'decode')
   self.pending=ctx
   try:
    ctx['logits_descriptor']=meta(logits);ctx['logprobs_mode']=owner.logprobs_mode;assert owner.logprobs_mode=='raw_logprobs' and owner.trace_replay_state is None
    ret=original_call(owner,logits,batch,*a,**kw)
    assert ctx['counts']==dict(gumbel=1,topk=1);ctx['original_calls']=1;ctx['passed']=True;self.rows.append(ctx);return ret
   finally:self.pending=None;self.save()
  pairs=[(S,'__call__',original_call,call)]
  for name,kind in [('gumbel_sample','gumbel'),('compute_topk_scores','topk')]:
   fn=getattr(module,name)
   def wrap(*a,_fn=fn,_kind=kind,**kw):return self.invoke(_kind,_fn,a,kw)
   pairs.append((module,name,fn,wrap))
  for obj,name,old,new in pairs:self.bindings.append((obj,name,old,new));setattr(obj,name,new)
 def save(self):
  rows=[{k:v for k,v in x.items() if k!='sampled'} for x in self.rows]
  r=dict(complete=len(rows)==16 and len(self.samples)==4 and all(x['passed'] for x in rows),rows=rows,samples=self.samples,observers_removed=self.closed,source=self.source,artifact_logical_bytes=self.bytes,scope='Native greedy and top5 source boundaries; first/last logical tensors only. No all16 logits capture/full backing proof or new operation support.')
  (self.out/'MANIFEST.json').write_text(json.dumps(r,indent=2)+'\n');return r
 def close(self):
  errors=[]
  for obj,name,old,new in reversed(self.bindings):
   try:assert getattr(obj,name) is new;setattr(obj,name,old)
   except BaseException as e:errors.append(repr(e))
  self.bindings=[];self.closed=not errors;assert not errors,errors
 def install(self):
  from vllm.v1.worker.gpu.sample import sampler
  self.install_classes(sampler.Sampler,sampler);return self.source
 def verify(self):assert self.save()['complete']
def install(model,out):
 assert not hasattr(model,ATTR);o=Observer(out);setattr(model,ATTR,o)
 try:return o.install()
 except BaseException:o.close();delattr(model,ATTR);raise
def finish(model):
 o=getattr(model,ATTR)
 try:o.close();o.verify();return o.save()
 finally:delattr(model,ATTR)
def close_only(model):
 o=getattr(model,ATTR,None)
 if o is None:return {'observer_absent':True}
 try:o.close();return {'observers_removed':True}
 finally:delattr(model,ATTR)
