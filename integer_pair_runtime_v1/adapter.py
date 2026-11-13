"""Endpoint-only integer pair. No payload oracle; original public APIs unchanged."""
from pathlib import Path
from embedding_int64_runtime_v2 import cast
from sampler_integer_runtime_v1 import integer
from lean26_gdn_finite_runtime_v1 import adapter as lean
from integer_pair_bridge_v1.bridge import Bridge
ATTR=lean.ATTR
PAIR='_integer_pair_runtime_v1'
ORDER=lean.ORDER
SPECS=lean.SPECS

def descriptor(x):
 return (tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.data_ptr(),x.is_neg(),x.is_conj())
def fresh(y,x,n,dtype,inputs):
 assert tuple(y.shape)==(n,) and y.stride()==(1,) and y.storage_offset()==0
 assert str(y.dtype)==dtype and y.device==x.device and not y.is_neg() and not y.is_conj()
 assert all(y.untyped_storage().data_ptr()!=a.untyped_storage().data_ptr() for a in inputs)

class Cast(cast.Cast):
 def latch(self,e):
  super().latch(e)
  p=getattr(self.model,PAIR,None)
  if p is not None:p.latch(e)
 def cast(self,x):
  from tileir_vllm_embedding_int64_v1.contract import validate
  from tileir_vllm_embedding_int64_v1.api import embedding_ids_long
  self.guard();n=validate(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.device.type,str(x.layout),x.requires_grad,x.is_neg(),x.is_conj())
  assert n==(1696 if len(self.rows)==0 else 32)
  before=descriptor(x)
  y=x.long() if self.mode=='shadow' else embedding_ids_long(x)
  fresh(y,x,n,'torch.int64',[x]);assert descriptor(x)==before
  self.rows.append(dict(index=len(self.rows),n=n,mode=self.mode,metadata_preserved=True,fresh=True,raw_compared=False,passed=True))
  return y
 def finish(self):
  assert self.active and self.failed is None and len(self.rows)==16
  assert [r['n'] for r in self.rows]==[1696]+[32]*15
  self.active=False
  if self.mode=='shadow':self.shadow_pass=True
  return dict(calls=16,mode=self.mode,rows=self.rows,native_calls=16 if self.mode=='shadow'else 0,tile_calls=16 if self.mode=='replace'else 0)

class Integer(integer.Integer):
 def fail(self,e):
  super().fail(e)
  p=getattr(self.model,PAIR,None)
  if p is not None:p.latch(e)
 def invoke(self,site,x,arg):
  from tileir_vllm_sampler_integer_v1.api import gather_integer_rows,new_sampled_ones
  from tileir_vllm_sampler_integer_v1.contract import gather_descriptor,validate_indices,ones_descriptor
  self.guard();assert site==('positions','input_ids','ones')[len(self.rows)%3]
  n=1696 if len(self.rows)//3==0 else 32
  assert x.device.type=='cuda' and not x.requires_grad
  if site=='ones':ones_descriptor(tuple(x.shape),str(x.dtype),arg);inputs=[x]
  else:
   assert tuple(x.shape)==(n,) and str(x.dtype)==('torch.int64'if site=='positions'else'torch.int32')
   assert arg.device==x.device and not arg.requires_grad and not x.is_neg() and not x.is_conj() and not arg.is_neg() and not arg.is_conj()
   size=gather_descriptor(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),tuple(arg.shape),tuple(arg.stride()),str(arg.dtype),arg.storage_offset())
   validate_indices(arg.cpu().tolist(),size);inputs=[x,arg]
  before=[descriptor(a) for a in inputs]
  if self.mode=='shadow':y=x.new_ones(arg)if site=='ones'else x[arg]
  else:y=new_sampled_ones(x,arg)if site=='ones'else gather_integer_rows(x,arg)
  fresh(y,x,32,str(x.dtype),inputs);assert before==[descriptor(a) for a in inputs]
  self.rows.append(dict(index=len(self.rows),site=site,mode=self.mode,metadata_preserved=True,fresh=True,raw_compared=False,passed=True))
  return y
 def finish(self):
  assert self.active and self.failed is None and len(self.rows)==48
  assert [r['site'] for r in self.rows]==list(('positions','input_ids','ones'))*16
  self.active=False
  if self.mode=='shadow':self.shadow_pass=True
  return dict(calls=48,mode=self.mode,rows=self.rows,site_counts={s:16 for s in ('positions','input_ids','ones')},native_calls=48 if self.mode=='shadow'else 0,tile_calls=48 if self.mode=='replace'else 0)

def install_pair(model,output):
 assert not any(hasattr(model,k) for k in (PAIR,cast.ATTR,integer.ATTR))
 root=Path(output);root.mkdir(exist_ok=False)
 a=Cast(model,root/'cast');b=Integer(model,root/'integer')
 p=Bridge(a,b,lambda:getattr(model,lean.ATTR,None));setattr(model,PAIR,p)
 setattr(model,cast.ATTR,a);setattr(model,integer.ATTR,b)
 p.bind_guards()
 try:return dict(cast=a.install(),integer=b.install(),scope='64 integer expressions; endpoint-only, not repeated raw proof')
 except BaseException as e:
  p.latch(e)
  try:p.close(lambda:dict(restored=True))
  except BaseException:pass
  raise

def install(model):return lean.install(model)
def start_shadow(model):return getattr(model,PAIR).start('shadow')
def finish_shadow(model):return getattr(model,PAIR).finish()
def start(model,mode='lean'):
 p=getattr(model,PAIR)
 try:p.start('replace');return lean.start(model,mode)
 except BaseException as e:p.latch(e);raise

def finish(model):
 p=getattr(model,PAIR)
 try:
  result=lean.finish(model);result['integer_pair']=p.finish();return result
 except BaseException as e:p.latch(e);raise

def close(model):return getattr(model,PAIR).close(lambda:lean.close(model))
def release(model):
 p=getattr(model,PAIR);p.require_release();lean.release(model);integer.release(model);cast.release(model);delattr(model,PAIR)
