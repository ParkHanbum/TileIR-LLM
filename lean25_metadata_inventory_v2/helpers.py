"""Scoped public helper metadata wrappers. Never change return values or tensor payloads."""
import functools,hashlib,importlib,inspect,json,traceback
from contextlib import contextmanager
from pathlib import Path

def metadata(torch,value,tensors,label='value',depth=0):
 if depth>8:raise RuntimeError('Metadata nesting cap')
 if isinstance(value,torch.Tensor):
  assert value.layout==torch.strided,'Sparse/opaque tensor metadata outside scope'
  storage=value.untyped_storage();start=storage.data_ptr();size=storage.nbytes()
  tensors.append((label,value,start,size))
  return dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),device=str(value.device),offset=value.storage_offset(),storage_bytes=size,requires_grad=value.requires_grad,is_conj=value.is_conj(),is_neg=value.is_neg())
 if isinstance(value,(list,tuple)):
  if len(value)>128:return dict(python_type=type(value).__name__,length=len(value),contents_omitted=True)
  return [metadata(torch,x,tensors,label+'.'+str(i),depth+1)for i,x in enumerate(value)]
 if isinstance(value,dict):
  assert len(value)<=128 and all(type(k)is str for k in value)
  return {k:metadata(torch,x,tensors,label+'.'+k,depth+1)for k,x in value.items()}
 return dict(python_type=type(value).__name__) # no scalar/list payload values/repr

def alias_rows(tensors):
 assert len(tensors)<=128,'Alias metadata cap'
 return [dict(left=a[0],right=b[0],same_object=a[1]is b[1],same_device=a[1].device==b[1].device,storage_overlap=a[1].device==b[1].device and a[2]<b[2]+b[3] and b[2]<a[2]+a[3],virtual_address_overlap=a[2]<b[2]+b[3] and b[2]<a[2]+a[3])for i,a in enumerate(tensors)for b in tensors[i+1:]]

def projected_owners(torch,owner,spec,tensors,label):
 projection=spec.get('typed_owner_projection')
 if not projection:return {}
 assert projection['container']=='compilation_config.static_forward_context' and projection['per_value_attribute']=='kv_cache'
 mapping=owner.compilation_config.static_forward_context
 assert type(mapping)is dict and len(mapping)<=128,'Only existing bounded public owner dictionary'
 out={}
 for name,module in mapping.items():
  assert type(name)is str
  if hasattr(module,'kv_cache'):
   value=module.kv_cache
   assert isinstance(value,(torch.Tensor,list,tuple))
   out[name]=metadata(torch,value,tensors,label+'.'+name)
 return out

def wrap_helper(torch,ledger,original,spec):
 sig=inspect.signature(original)
 @functools.wraps(original)
 def wrapper(*args,**kwargs):
  if not getattr(ledger,"active",True):return original(*args,**kwargs)
  bound=sig.bind(*args,**kwargs);bound.apply_defaults();tensors=[]
  before={k:metadata(torch,v,tensors,k)for k,v in bound.arguments.items()}
  owner=bound.arguments.get('self')
  for path in spec.get('self_attrs',[]):
   value=owner
   for bit in path.split('.'):
    if value is None or not hasattr(value,bit):value=None;break
    value=getattr(value,bit)
   before['self.'+path]=metadata(torch,value,tensors,'self.'+path)
  before['typed_owner_projection']=projected_owners(torch,owner,spec,tensors,'before.kv_owner')
  before['aliases']=alias_rows(tensors)
  frames=[dict(file=x.filename,line=x.lineno,function=x.name)for x in traceback.extract_stack(limit=24)[:-1]][-12:]
  idx=ledger.site('helper::'+spec['label'],before,frames)
  try:
   with torch.profiler.record_function('python_helper_site::'+str(idx)):result=original(*args,**kwargs)
   after_tensors=list(tensors);out=metadata(torch,result,after_tensors,'return');after=dict(output=out,aliases=alias_rows(after_tensors))
   # Re-observe descriptors only; this is not a mutation/value preservation proof.
   after['inputs']={k:metadata(torch,v,[],k)for k,v in bound.arguments.items()}
   for path in spec.get('self_attrs',[]):
    value=owner
    for bit in path.split('.'):
     if value is None or not hasattr(value,bit):value=None;break
     value=getattr(value,bit)
    after['inputs']['self.'+path]=metadata(torch,value,after_tensors,'after.self.'+path)
   after['typed_owner_projection']=projected_owners(torch,owner,spec,after_tensors,'after.kv_owner')
   after['aliases']=alias_rows(after_tensors)
   encoded=json.dumps(after,sort_keys=True);variants=ledger.rows[idx].setdefault('return_metadata',{})
   if encoded not in variants:
    if ledger.bytes+len(encoded.encode())>ledger.max_bytes:raise RuntimeError('Post-call metadata cap; original already executed')
    ledger.bytes+=len(encoded.encode());variants[encoded]=0
   variants[encoded]+=1;ledger.rows[idx]['returns']+=1
   return result
  except BaseException:
   ledger.failed=True;ledger.rows[idx]['errors']+=1;raise
 return wrapper

@contextmanager
def helper_observers(ledger,root,spec_filename='HELPER-SPECS.json'):
 import torch
 doc=json.loads((Path(root)/spec_filename).read_text());assert doc['complete']
 installed=[]
 try:
  for spec in doc['helpers']:
   assert hashlib.sha256(Path(spec['source_path']).read_bytes()).hexdigest()==spec['source_sha256']
   obj=importlib.import_module(spec['module'])
   for name in spec.get('owner', '').split('.'):
    if name:obj=getattr(obj,name)
   name=spec['attribute'];original=getattr(obj,name);assert callable(original)
   assert list(inspect.signature(original).parameters)==spec['parameter_names']
   had=name in vars(obj);prior=vars(obj).get(name);new=wrap_helper(torch,ledger,original,spec)
   installed.append((obj,name,original,had,prior,new));setattr(obj,name,new)
  yield
  totals={h['label']:sum(row['calls'] for row in ledger.rows if row['op']=='helper::'+h['label']) for h in doc['helpers']}
  for h in doc['helpers']:
   n=totals[h['label']];assert n>=0
   if h.get('expected_calls') is not None:assert n==h['expected_calls'],('Helper census',h['label'],n)
  ledger.helper_counts=totals
 finally:
  errors=[]
  for obj,name,original,had,prior,new in reversed(installed):
   try:
    assert getattr(obj,name)is new,'Foreign helper binding'
    if had:setattr(obj,name,prior)
    else:delattr(obj,name)
    assert getattr(obj,name)is original
   except BaseException as e:errors.append(repr(e))
  ledger.helpers_restored=not errors
  if errors:ledger.failed=True;raise RuntimeError('Helper restoration HOLD: '+str(errors))
