"""Three owned sampler expressions; correctness instrumentation only."""
import inspect,textwrap
from pathlib import Path
R=Path(__file__).parent
ATTR='_sampler_integer_runtime_v1'
HELPERS=('get_num_nans','compute_topk_scores','get_num_sampled_and_rejected')
class Integer:
 def __init__(self,model,output):
  self.model=model;self.output=Path(output);self.output.mkdir(exist_ok=False);self.active=False;self.failed=None;self.rows=[];self.mode=None;self.shadow_pass=False;self.owner=None;self.wrapped=None
 def fail(self,e):
  if self.failed is None:self.failed=repr(e)
  from lean26_gdn_finite_runtime_v1.adapter import ATTR as LEAN
  controller=getattr(self.model,LEAN,None)
  if controller is not None:controller.latch('sampler_integer',e)
 def guard(self):
  assert self.active and self.failed is None and len(self.rows)<48,'Integer expressions inactive, failed or exhausted'
  from lean26_gdn_finite_runtime_v1.adapter import ATTR as LEAN
  controller=getattr(self.model,LEAN,None)
  assert controller is None or controller.failed is None,'Another component failed'
 def install(self):
  from vllm.v1.worker.gpu.sample import sampler
  cls=sampler.Sampler;old=cls.__call__;source=textwrap.dedent(inspect.getsource(old))
  assert source.strip()==(R/'call.original.py.txt').read_text().strip()
  candidate=(R/'call.candidate.py.txt').read_text()
  expected=source.replace('input_batch.positions[input_batch.logits_indices]',"_integer_gather('positions',input_batch.positions,input_batch.logits_indices)").replace('input_batch.input_ids[input_batch.logits_indices]',"_integer_gather('input_ids',input_batch.input_ids,input_batch.logits_indices)").replace('input_batch.seq_lens.new_ones(input_batch.num_reqs)','_new_sampled_ones(input_batch.seq_lens,input_batch.num_reqs)')
  assert candidate==expected
  ns=dict(old.__globals__);ns['_integer_gather']=self.gather;ns['_new_sampled_ones']=self.ones
  self.initial_helpers={key:getattr(sampler,key)for key in HELPERS}
  assert all(old.__globals__.get(key)is value for key,value in self.initial_helpers.items()),'Original helper provenance mismatch'
  for key in HELPERS:
   def delegate(*a,_key=key,**k):return getattr(sampler,_key)(*a,**k)
   ns[key]=delegate
  exec(compile(candidate,str(R/'call.candidate.py.txt'),'exec'),ns);body=ns['__call__']
  self.owner=cls;self.old=old;self.had='__call__'in vars(cls);self.prior=vars(cls).get('__call__');self.module=sampler
  def wrapped(owner,*a,**k):
   try:
    self.guard();before=len(self.rows);assert before%3==0
    out=body(owner,*a,**k);assert len(self.rows)==before+3;return out
   except BaseException as e:self.fail(e);raise
  self.wrapped=wrapped;cls.__call__=wrapped
  return dict(expected_calls=48,helpers_live=HELPERS,scope='Three expressions only; installed inside later helper wrappers')
 def start(self,mode):
  assert not self.active and self.failed is None and mode in ('shadow','replace')and(mode!='replace'or self.shadow_pass)
  self.active=True;self.mode=mode;self.rows=[]
 def gather(self,site,x,idx):return self.invoke(site,x,idx)
 def ones(self,x,n):return self.invoke('ones',x,n)
 def invoke(self,site,x,arg):
  import torch
  from tileir_vllm_sampler_integer_v1.api import gather_integer_rows,new_sampled_ones
  from tileir_vllm_sampler_integer_v1.contract import gather_descriptor,validate_indices,ones_descriptor
  self.guard();assert site==('positions','input_ids','ones')[len(self.rows)%3]
  n=1696 if len(self.rows)//3==0 else 32
  assert x.device.type=='cuda'and not x.requires_grad
  if site=='ones':ones_descriptor(tuple(x.shape),str(x.dtype),arg);inputs=[x]
  else:
   assert tuple(x.shape)==(n,)and x.dtype==(torch.int64 if site=='positions'else torch.int32)
   assert arg.device==x.device and not arg.requires_grad and not x.is_neg()and not x.is_conj()and not arg.is_neg()and not arg.is_conj()
   size=gather_descriptor(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),tuple(arg.shape),tuple(arg.stride()),str(arg.dtype),arg.storage_offset());validate_indices(arg.cpu().tolist(),size);inputs=[x,arg]
  def meta(t):return(tuple(t.shape),tuple(t.stride()),str(t.dtype),t.storage_offset(),t.data_ptr(),t.is_neg(),t.is_conj())
  def snap(t):
   s=t.untyped_storage();assert s.nbytes()<=1048576
   return torch.empty(0,device=t.device,dtype=torch.uint8).set_(s,0,(s.nbytes(),),(1,)).cpu()
  before=[snap(t)for t in inputs];metadata=[meta(t)for t in inputs];native=None;candidate=None
  try:
   if self.mode=='shadow':native=x.new_ones(arg)if site=='ones'else x[arg]
   candidate=new_sampled_ones(x,arg)if site=='ones'else gather_integer_rows(x,arg)
   assert tuple(candidate.shape)==(32,)and candidate.stride()==(1,)and candidate.storage_offset()==0 and candidate.dtype==x.dtype and candidate.device==x.device
   assert all(candidate.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr()for t in inputs)
   if native is not None:assert torch.equal(native.view(torch.uint8),candidate.view(torch.uint8))and candidate.data_ptr()!=native.data_ptr()
   assert all(torch.equal(b,snap(t))and m==meta(t)for b,m,t in zip(before,metadata,inputs))
   self.rows.append(dict(index=len(self.rows),site=site,mode=self.mode,raw_compared=native is not None,input_owner_preserved=True,metadata_preserved=True,fresh=True,returned_native_identity=native is not None,passed=True))
   return native if native is not None else candidate
  except BaseException as e:
   self.fail(e);p=self.output/'FIRST-FAILURE.pt'
   if not p.exists():torch.save(dict(site=site,mode=self.mode,index=len(self.rows),input=[t.cpu()for t in inputs],owner_before=before,native=None if native is None else native.cpu(),candidate=None if candidate is None else candidate.cpu(),error=repr(e)),p)
   raise
 def finish(self):
  assert self.active and self.failed is None and len(self.rows)==48
  self.active=False
  if self.mode=='shadow':self.shadow_pass=True
  return dict(mode=self.mode,calls=48,rows=self.rows,native_integer_calls=48 if self.mode=='shadow'else 0,tile_integer_calls=48,shadow_pass=self.shadow_pass,site_counts={s:sum(x['site']==s for x in self.rows)for s in ('positions','input_ids','ones')})
 def close(self):
  if self.owner is not None and self.wrapped is not None:
   assert self.owner.__call__ is self.wrapped,'Close outer wrappers before integer source owner'
   assert all(getattr(self.module,key)is value for key,value in self.initial_helpers.items()),'Outer helper restoration incomplete'
   if self.had:self.owner.__call__=self.prior
   else:delattr(self.owner,'__call__')
   assert self.owner.__call__ is self.old;self.wrapped=None
  self.active=False;return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR);s=Integer(model,output);setattr(model,ATTR,s)
 try:return s.install()
 except BaseException:s.close();delattr(model,ATTR);raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 s=getattr(model,ATTR,None);return s.close()if s is not None else dict(restored=True)
def release(model):
 s=getattr(model,ATTR,None)
 if s is not None:assert s.wrapped is None;delattr(model,ATTR)
