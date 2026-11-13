"""Owned forward cast-only correctness adapter; no global Tensor patch."""
import inspect,textwrap,json
from pathlib import Path
R=Path(__file__).parent
ATTR='_embedding_int64_cast_v1'
class Cast:
 def __init__(self,model,output):
  self.model=model;self.output=Path(output);self.output.mkdir(exist_ok=False);self.active=False;self.failed=None;self.mode=None;self.rows=[];self.shadow_pass=False;self.owner=None;self.old=None;self.wrapped=None
 def latch(self,e):
  if self.failed is None:self.failed=repr(e)
 def guard(self):assert self.active and self.failed is None and len(self.rows)<16,'Cast outside active request, after failure, or beyond16'
 def install(self):
  from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
  found=[m for n,m in self.model.named_modules()if n.endswith('embed_tokens')and type(m)is VocabParallelEmbedding];assert len(found)==1
  owner=found[0];assert owner.tp_size==1
  original=owner.forward;source=textwrap.dedent(inspect.getsource(original));assert source.strip()==(R/'forward.original.py.txt').read_text().strip()
  candidate=(R/'forward.candidate.py.txt').read_text();assert candidate==source.replace('input_.long()','_cast_ids(input_)')
  ns=dict(original.__func__.__globals__);ns['_cast_ids']=self.cast;exec(compile(candidate,str(R/'forward.candidate.py.txt'),'exec'),ns);body=ns['forward']
  self.owner=owner;self.had='forward'in vars(owner);self.prior=vars(owner).get('forward');self.old=original
  def wrapped(input_):
   try:self.guard();assert owner.tp_size==1;return body(owner,input_)
   except BaseException as e:self.latch(e);raise
  self.wrapped=wrapped;owner.forward=wrapped
  return dict(scope='Owned TP1 forward long expression only; current quant_method.embedding delegated',expected_calls=16)
 def start(self,mode):
  assert not self.active and self.failed is None and mode in ('shadow','replace')
  assert mode!='replace'or self.shadow_pass
  self.mode=mode;self.rows=[];self.active=True
 def cast(self,x):
  import torch
  from tileir_vllm_embedding_int64_v1.contract import validate
  from tileir_vllm_embedding_int64_v1.api import embedding_ids_long
  self.guard();n=validate(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.device.type,str(x.layout),x.requires_grad,x.is_neg(),x.is_conj())
  assert n==(1696 if len(self.rows)==0 else 32)
  storage=x.untyped_storage();assert storage.nbytes()<=1048576
  def snapshot():return torch.empty(0,dtype=torch.uint8,device=x.device).set_(storage,0,(storage.nbytes(),),(1,)).cpu()
  before=snapshot();metadata=(x.data_ptr(),x.storage_offset(),tuple(x.shape),tuple(x.stride()),x.is_neg(),x.is_conj());native=None;candidate=None
  try:
   if self.mode=='shadow':native=x.long()
   candidate=embedding_ids_long(x)
   def check(y):
    assert y.dtype==torch.int64 and tuple(y.shape)==(n,)and y.stride()==(1,)and y.storage_offset()==0 and y.device==x.device
    assert y.untyped_storage().data_ptr()!=storage.data_ptr()
   check(candidate)
   if native is not None:
    check(native);assert native.untyped_storage().data_ptr()!=candidate.untyped_storage().data_ptr()
    assert torch.equal(native.view(torch.uint8),candidate.view(torch.uint8)),'Cast raw mismatch'
   assert torch.equal(before,snapshot()),'Input full owner mutated'
   assert metadata==(x.data_ptr(),x.storage_offset(),tuple(x.shape),tuple(x.stride()),x.is_neg(),x.is_conj())
   self.rows.append(dict(index=len(self.rows),n=n,mode=self.mode,raw_compared=native is not None,input_owner_preserved=True,metadata_preserved=True,fresh=True,returned_native_identity=native is not None,passed=True))
   return native if native is not None else candidate
  except BaseException as e:
   self.latch(e)
   bank=self.output/'FIRST-FAILURE.pt'
   if not bank.exists():torch.save(dict(mode=self.mode,index=len(self.rows),input=x.cpu(),input_owner_before=before,native=None if native is None else native.cpu(),candidate=None if candidate is None else candidate.cpu(),error=repr(e)),bank)
   raise
 def finish(self):
  assert self.active and self.failed is None and len(self.rows)==16 and [x['n']for x in self.rows]==[1696]+[32]*15
  self.active=False
  if self.mode=='shadow':self.shadow_pass=True
  return dict(mode=self.mode,calls=16,rows=self.rows,native_cast_calls=16 if self.mode=='shadow'else 0,tile_cast_calls=16,shadow_pass=self.shadow_pass,scope='Cast-only rows; downstream embedding independently delegated')
 def close(self):
  if self.owner is not None and self.wrapped is not None:
   assert self.owner.forward is self.wrapped
   if self.had:self.owner.forward=self.prior
   else:delattr(self.owner,'forward')
   assert self.owner.forward==self.old;self.wrapped=None
  self.active=False;return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR);s=Cast(model,output);setattr(model,ATTR,s)
 try:return s.install()
 except BaseException:s.close();delattr(model,ATTR);raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 s=getattr(model,ATTR,None)
 if s is None:return dict(restored=True)
 return s.close()
def release(model):
 s=getattr(model,ATTR,None)
 if s is not None:assert s.wrapped is None;delattr(model,ATTR)
