"""Native-only TP1 embedding method observer; selected row evidence, not full table."""
from pathlib import Path
import sys,json,hashlib,functools
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from norm_runtime_contract_capture_v1.observer import metadata,identity,aliases
ATTR='_tp1_embedding_capture_v1'
CAP=128*1024*1024
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def version(t):
 try:return t._version
 except RuntimeError as e:
  if 'Inference tensors do not track version counter' in str(e):return 'unavailable_inference_tensor'
  raise
def meta(t):
 d=metadata(t);d.update(requires_grad=bool(t.requires_grad),is_leaf=bool(t.is_leaf),python_type=type(t).__module__+'.'+type(t).__name__,version=version(t));return d
def phase(n):
 if n==1696:return 'prefill'
 if n==32:return 'decode'
 raise AssertionError('Unexpected embedding token count')
def digest(t):
 import torch
 return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
class Observer:
 def __init__(self,out):
  self.out=Path(out);self.out.mkdir(exist_ok=False);self.rows=[];self.samples=[];self.binding=None;self.pending=False;self.bytes=0;self.closed=False
 def tensor(self,t,label):
  import torch
  c=t.detach().cpu().contiguous().clone();self.bytes+=c.numel()*c.element_size();assert self.bytes<=CAP
  f=self.out/(label+'.pt');assert not f.exists();torch.save(c,f)
  return dict(path=str(f),sha256=sha(f),logical_sha256=digest(c),metadata=meta(t))
 def save(self):
  counts={s:sum(x['phase']==s for x in self.rows)for s in ('prefill','decode')}
  r=dict(schema='tp1-embedding-native-v1',complete=counts=={'prefill':1,'decode':15}and len(self.samples)==2 and not self.pending and all(x['passed']for x in self.rows),counts=counts,rows=self.rows,samples=self.samples,observers_removed=self.closed,artifact_logical_bytes=self.bytes,source_pins=json.loads(Path(__file__).with_name('PINS.json').read_text()),weight_linkage='Actual live Parameter identity and selected-row bytes only. Checkpoint index/config hashes are provenance, not whole live table equality proof.')
  (self.out/'MANIFEST.json').write_text(json.dumps(r,indent=2)+'\n');return r
 def install(self,model):
  from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding,UnquantizedEmbeddingMethod
  found=[(n,m)for n,m in model.named_modules()if n.endswith('embed_tokens')and type(m)is VocabParallelEmbedding];assert len(found)==1
  name,m=found[0];assert m.tp_size==1 and type(m.quant_method)is UnquantizedEmbeddingMethod and tuple(m.weight.shape)==(248320,2560)
  obj=m.quant_method;original=obj.embedding;assert original.__func__ is UnquantizedEmbeddingMethod.embedding
  self.module=name;self.source=identity(original);self.weight_id=id(m.weight)
  @functools.wraps(original)
  def wrapped(layer,ids):
   if layer is not m:return original(layer,ids)
   import torch
   assert not self.pending and id(layer.weight)==self.weight_id;self.pending=True
   try:
    assert ids.ndim==1 and ids.dtype==torch.int64
    ph=phase(ids.numel());assert len(self.rows)<16
    v0=version(m.weight);id_hash=digest(ids);idcpu=ids.detach().cpu().clone();assert bool(((idcpu>=0)&(idcpu<248320)).all())
    selected=not any(x['phase']==ph for x in self.samples)
    row=dict(phase=ph,original_calls=0,source=self.source,input=meta(ids),weight=meta(m.weight))
    sample=None
    if selected:
     unique,inverse=torch.unique(ids,sorted=True,return_inverse=True)
     rows=m.weight.index_select(0,unique).detach();before=digest(rows)
     sample=dict(phase=ph,input=self.tensor(ids,ph+'_ids'),unique_ids=self.tensor(unique,ph+'_unique'),inverse=self.tensor(inverse,ph+'_inverse'),selected_weight_rows=self.tensor(rows,ph+'_rows'),weight_descriptor=meta(m.weight))
    result=original(layer,ids);row['original_calls']=1
    assert tuple(result.shape)==(ids.numel(),2560)and result.dtype==torch.bfloat16
    row.update(ids_preserved=digest(ids)==id_hash,weight_version_before=v0,weight_version_after=version(m.weight),output=meta(result),aliases=aliases({'ids':ids,'weight':m.weight,'output':result}))
    row['fresh']=all(not x['storage_overlap']for x in row['aliases'].values());row['passed']=row['ids_preserved']and row['fresh']and row['weight_version_before']==row['weight_version_after']
    if selected:
     sample['output']=self.tensor(result,ph+'_output');sample['selected_rows_preserved']=digest(m.weight.index_select(0,unique))==before
     sample['raw_rows_equal']=digest(result)==digest(rows.index_select(0,inverse));sample['finite']=bool(torch.isfinite(result).all())
     row['passed'] &= sample['selected_rows_preserved']and sample['raw_rows_equal']and sample['finite'];self.samples.append(sample)
    self.rows.append(row);assert row['passed'];return result
   finally:self.pending=False;self.save()
  had='embedding'in obj.__dict__;prior=obj.__dict__.get('embedding');self.binding=(obj,had,prior,wrapped);obj.embedding=wrapped;return dict(module=name,source=self.source)
 def close(self):
  if self.binding:
   obj,had,prior,wrap=self.binding;assert obj.__dict__.get('embedding')is wrap
   if had:obj.embedding=prior
   else:delattr(obj,'embedding')
   self.binding=None
  self.closed=True
 def verify(self):assert len(self.rows)==16 and self.save()['complete']
def install(model,out):
 assert not hasattr(model,ATTR);o=Observer(out);setattr(model,ATTR,o)
 try:return o.install(model)
 except BaseException:o.close();delattr(model,ATTR);raise
def finish(model):
 o=getattr(model,ATTR,None)
 if o is None:return {'observer_absent':True}
 try:o.close();o.verify();return o.save()
 finally:delattr(model,ATTR)
def close_only(model):
 o=getattr(model,ATTR,None)
 if o is None:return {'observer_absent':True}
 try:o.close();return {'observers_removed':True}
 finally:delattr(model,ATTR)
