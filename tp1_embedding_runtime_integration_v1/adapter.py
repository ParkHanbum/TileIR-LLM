"""Actual live TP1 embedding, selected-row proof only; no full table clone."""
from pathlib import Path
from collections import Counter
import json,hashlib
ATTR='_tp1_embedding_functional_v1'
def census(rows):return len(rows)==16 and Counter(r['phase'] for r in rows)==Counter(prefill=1,decode=15)
def overlap(a,b):
 x,y=a.untyped_storage(),b.untyped_storage()
 return max(x.data_ptr(),y.data_ptr())<min(x.data_ptr()+x.nbytes(),y.data_ptr()+y.nbytes())
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.binding=None;self.shadow_pass=False;self.mode='unstarted';self.rows=[];self.native_gate_calls=0;self.tileir_gate_calls=0;self.first_mismatch=None
 def install(self,model):
  from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding,UnquantizedEmbeddingMethod
  matches=[(n,m) for n,m in model.named_modules() if n.endswith('embed_tokens') and type(m)is VocabParallelEmbedding]
  assert len(matches)==1
  self.name,self.layer=matches[0];m=self.layer;q=m.quant_method
  assert m.tp_size==1 and type(q)is UnquantizedEmbeddingMethod
  self.weight=m.weight;assert tuple(self.weight.shape)==(248320,2560)
  orig=q.embedding;had='embedding' in q.__dict__;prior=q.__dict__.get('embedding')
  def wrapper(layer,ids):
   assert layer is self.layer,'Unexpected shared quant method owner'
   return self.apply(layer,ids,orig)
  self.binding=(q,had,prior,wrapper);q.embedding=wrapper
  return dict(module=self.name,boundary='UnquantizedEmbeddingMethod.embedding',weight_requires_grad=self.weight.requires_grad,weight_id=id(self.weight))
 def apply(self,layer,ids,native):
  import torch
  from tileir_vllm_tp1_embedding_v1 import embedding
  assert self.mode in ('shadow','replace') and (self.mode!='replace' or self.shadow_pass)
  assert layer.weight is self.weight and tuple(ids.shape) in ((1696,),(32,)) and ids.dtype==torch.int64 and ids.stride()==(1,) and ids.storage_offset()==0
  assert ids.untyped_storage().nbytes()==ids.numel()*ids.element_size()
  w=self.weight;version=w._version
  before=ids.detach().clone();unique,inverse=torch.unique(ids,sorted=True,return_inverse=True)
  selected=w.detach().index_select(0,unique).clone();assert bool(torch.isfinite(selected).all())
  expected=selected.index_select(0,inverse)
  if self.mode=='shadow':out=native(layer,ids);self.native_gate_calls+=1
  candidate=embedding(ids.detach(),w.detach());self.tileir_gate_calls+=1
  if self.mode=='replace':out=candidate
  raw=lambda a,b:bool(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))
  preserved=raw(before,ids) and raw(selected,w.detach().index_select(0,unique)) and layer.weight is w and w._version==version
  finite=bool(torch.isfinite(out).all()) and bool(torch.isfinite(candidate).all())
  layout=out.shape==candidate.shape==expected.shape and out.dtype==candidate.dtype==torch.bfloat16 and out.stride()==candidate.stride()==(2560,1) and out.storage_offset()==candidate.storage_offset()==0
  alias=not any(overlap(y,t) for y in (out,candidate) for t in (ids,w)) and (self.mode!='shadow' or not overlap(out,candidate))
  equal=raw(out,candidate) and raw(candidate,expected);strict=bool(torch.allclose(out.float(),candidate.float(),atol=1e-5,rtol=1e-5))
  passed=preserved and finite and layout and alias and equal and strict
  row=dict(phase='prefill' if len(ids)==1696 else 'decode',shape=list(ids.shape),mode=self.mode,raw_equal=equal,strict=strict,finite=finite,input_and_selected_rows_preserved=preserved,layout=layout,fresh_nonalias=alias,weight_version=version,weight_requires_grad=w.requires_grad,selected_rows=len(unique),passed=passed)
  self.rows.append(row)
  if not passed:
   bank=self.output/'first_mismatch';bank.mkdir();arts={}
   for key,t in [('ids',before),('unique_ids',unique),('selected_rows',selected),('native_or_actual',out),('candidate',candidate)]:
    p=bank/(key+'.pt');torch.save(t.detach().cpu(),p);arts[key]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
   self.first_mismatch=dict(row=row,artifacts=arts)
  self.save();assert passed,'Embedding raw/contract HOLD'
  return out
 def start(self,mode):
  assert mode in ('shadow','replace') and (mode!='replace' or self.shadow_pass)
  self.mode=mode;self.rows=[];self.native_gate_calls=self.tileir_gate_calls=0;return dict(mode=mode)
 def summary(self):return dict(mode=self.mode,rows=self.rows,supported_calls=len(self.rows),native_gate_calls=self.native_gate_calls,tileir_gate_calls=self.tileir_gate_calls,shadow_pass=self.shadow_pass,first_mismatch=self.first_mismatch,supported_vendor_fallback_calls=0,weight_proof='Parameter identity/version and accessed selected rows only; no full live table invariance proof')
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2))
 def finish(self):
  assert census(self.rows) and all(r['passed'] for r in self.rows) and self.tileir_gate_calls==16 and self.native_gate_calls==(16 if self.mode=='shadow' else 0)
  if self.mode=='shadow':self.shadow_pass=True
  self.save();return self.summary()
 def close(self):
  if self.binding is not None:
   q,had,prior,wrapper=self.binding;assert q.__dict__.get('embedding') is wrapper
   if had:q.embedding=prior
   else:delattr(q,'embedding')
   self.binding=None
  return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR);a=Adapter(output)
 try:r=a.install(model);setattr(model,ATTR,a);return r
 except BaseException:a.close();raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 r=a.close();delattr(model,ATTR);return r
