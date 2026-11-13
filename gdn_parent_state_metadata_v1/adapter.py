"""Native-only parent-state metadata observer. No owner values copied."""
from pathlib import Path
import json,functools,inspect
from norm_runtime_contract_capture_v1.observer import metadata,identity,aliases
from gdn_post_conv_runtime_capture_v1.observer import phase_meta
ATTR='_gdn_parent_state_metadata_v1'
ORDER=('gather','invert','mask','core_enter','core_exit','scatter')
def extent(shape,stride,offset,item,storage):
 assert len(shape)==len(stride) and all(type(x)is int and x>0 for x in shape) and all(type(x)is int and x>=0 for x in stride) and offset>=0
 logical=item
 for x in shape:logical*=x
 end=(offset+sum((d-1)*s for d,s in zip(shape,stride))+1)*item
 assert logical<=1024**3 and end<=storage<=32*1024**3 and end-offset*item<=1024**3
 return dict(logical_bytes=logical,storage_extent_bytes=end-offset*item,storage_offset_bytes=offset*item)
def values_contract(indices,mask,rows):
 assert len(indices)==len(mask)==32 and all(type(x)is int for x in indices) and all(type(x)is bool for x in mask)
 assert all(-rows<=x<rows for x in indices)
 normalized=[x%rows for x in indices]
 return dict(indices=indices,normalized_indices=normalized,unique=len(set(normalized))==32,has_initial_state=mask,in_bounds=True)
def advance(row,label):
 assert len(row['order'])<len(ORDER) and label==ORDER[len(row['order'])],(row['order'],label)
 row['order'].append(label)
class Adapter:
 def __init__(self,output):
  self.out=Path(output);self.out.mkdir(exist_ok=False);self.bindings=[];self.modules={};self.mode=None;self.rows=[];self.failed=None;self.active=None;self.baseline_pass=False
 def guard(self):assert self.failed is None,('Latched first failure',self.failed)
 def fail(self,error):
  if self.failed is None:self.failed=dict(error=repr(error),row=None if self.active is None else self.active['row'])
  self.save()
 def save(self):
  r=dict(mode=self.mode,rows=self.rows,failed=self.failed,native_only=True,owner_payload_bytes=0,metadata_only=True)
  (self.out/((self.mode or 'install')+'-PROGRESS.json')).write_text(json.dumps(r,indent=2));return r
 def small(self,t,name,owner):
  import torch
  assert t.is_cuda and t.device==owner.device and tuple(t.shape)==(32,) and t.stride()[0]>=0 and t.numel()*t.element_size()<=256
  assert t.untyped_storage().nbytes()<=16*1024*1024
  assert t.dtype in ((torch.int32,torch.int64)if name=='indices'else(torch.bool,))
  return t.detach().cpu().tolist()
 def observe(self,func,args,kwargs):
  self.guard();ctx=self.active;owner=ctx['owner'];row=ctx['row'];op=str(func);label=None
  if op=='aten.index.Tensor' and args[0] is owner:label='gather';assert len(args[1])==1 and args[1][0] is ctx['indices']
  elif op=='aten.bitwise_not.default' and args[0] is ctx['mask']:label='invert'
  elif op in ('aten.index_put_.default','aten._index_put_impl_.default','aten.masked_fill_.Scalar') and args[0] is ctx.get('initial'):label='mask'
  elif op in ('aten.index_put_.default','aten._index_put_impl_.default') and args[0] is owner:label='scatter'
  if label:
   advance(row,label)
   if label in ('mask','scatter'):
    if op=='aten.masked_fill_.Scalar':assert args[1].untyped_storage().data_ptr()==ctx['inverse_mask'].untyped_storage().data_ptr() and args[2]==0
    else:
     assert len(args[1])==1
     if label=='mask':
      assert args[1][0] is ctx['inverse_mask'];value=args[2]
      if hasattr(value,'numel'):
       assert value.numel()==1 and value.element_size()<=8 and value.untyped_storage().nbytes()<=8;assert value.item()==0
      else:assert value==0
     else:assert args[1][0] is ctx['indices'] and args[2] is ctx['final']
     assert not (args[3] if len(args)>3 else kwargs.get('accumulate',False))
   row.setdefault('dispatch',[]).append(dict(label=label,operator=op))
  result=func(*args,**kwargs) # Original operator exactly once, same return.
  if label=='gather':
   ctx['initial']=result;row['gathered']=metadata(result);assert tuple(result.shape)==(32,32,128,128) and tuple(result.stride())==(524288,16384,128,1) and result.storage_offset()==0 and result.untyped_storage().nbytes()==64*1024*1024
   assert not any(x['storage_overlap'] for x in aliases({'owner':owner,'initial':result}).values())
  elif label=='invert':
   assert tuple(result.shape)==(32,) and result.element_size()==1 and result.untyped_storage().nbytes()<=256
   ctx['inverse_mask']=result;row['inverse_mask']=metadata(result)
  elif label in ('mask','scatter'):assert result is (ctx['initial']if label=='mask'else owner)
  return result
 def install(self,model):
  import torch
  from torch.utils._python_dispatch import TorchDispatchMode
  from vllm.forward_context import get_forward_context
  observer=self
  class Mode(TorchDispatchMode):
   def __torch_dispatch__(self,func,types,args=(),kwargs=None):return observer.observe(func,args,kwargs or {})
  def bind(obj,key,wrapper):
   had=key in obj.__dict__;old=obj.__dict__.get(key);self.bindings.append((obj,key,had,old,wrapper));setattr(obj,key,wrapper)
  for name,m in model.named_modules():
   if not name.endswith('.linear_attn') or not hasattr(m,'chunk_gated_delta_rule'):continue
   original=m._forward_core;core=m.chunk_gated_delta_rule;native=core._forward_method;sig=inspect.signature(native)
   self.modules[name]=dict(parent=identity(original),core=identity(native));assert native.__name__=='forward_native'
   @functools.wraps(original)
   def parent(*args,_m=m,_name=name,_orig=original,**kwargs):
    self.guard()
    try:
     ctx=get_forward_context();md=ctx.attn_metadata[_m.prefix];phase=phase_meta(md)
     if phase['phase']!='prefill':return _orig(*args,**kwargs)
     assert self.active is None and not phase['speculative'] and phase['num_actual_tokens']==1696 and len(self.rows)<24
     owner=_m.kv_cache[1];assert owner.is_cuda and owner.dtype==torch.float32 and tuple(owner.shape[1:])==(32,128,128) and tuple(owner.stride())==(540672,16384,128,1)
     bounds=extent(tuple(owner.shape),tuple(owner.stride()),owner.storage_offset(),owner.element_size(),owner.untyped_storage().nbytes())
     indices=md.prefill_state_indices;mask=md.prefill_has_initial_state;iv=self.small(indices,'indices',owner);mv=self.small(mask,'mask',owner)
     row=dict(layer=_name,phase=phase,owner=metadata(owner),owner_extent=bounds,indices=metadata(indices),mask=metadata(mask),values=values_contract(iv,mv,owner.shape[0]),order=[],passed=False);self.rows.append(row)
     self.active=dict(row=row,owner=owner,indices=indices,mask=mask);self.save()
     with Mode():ret=_orig(*args,**kwargs)
     assert row['order']==list(ORDER);assert self.small(indices,'indices',owner)==iv and self.small(mask,'mask',owner)==mv
     row['passed']=True;self.save();return ret
    except BaseException as e:self.fail(e);raise
    finally:self.active=None
   @functools.wraps(native)
   def corewrap(*args,_orig=native,_sig=sig,**kwargs):
    self.guard()
    try:
     assert self.active is not None;ctx=self.active;b=_sig.bind(*args,**kwargs);b.apply_defaults();assert b.arguments['initial_state'] is ctx['initial'];advance(ctx['row'],'core_enter')
     # Dispatch observer remains active but ignores unrelated core operators.
     ret=_orig(*args,**kwargs);assert isinstance(ret,tuple) and len(ret)==2;ctx['final']=ret[1];advance(ctx['row'],'core_exit');ctx['row']['final']=metadata(ret[1]);ctx['row']['aliases']=aliases({'owner':ctx['owner'],'initial':ctx['initial'],'final':ret[1]});return ret
    except BaseException as e:self.fail(e);raise
   bind(m,'_forward_core',parent);bind(core,'_forward_method',corewrap)
  assert len(self.modules)==24;return dict(modules=self.modules,native_only=True)
 def start(self,mode):
  self.guard()
  try:
   assert self.active is None and mode in ('shadow','replace') and (mode!='replace'or self.baseline_pass);self.mode=mode;self.rows=[];return dict(native_only=True,mode=mode)
  except BaseException as e:self.fail(e);raise
 def finish(self):
  self.guard()
  try:assert self.active is None and len(self.rows)==24 and len({x['layer']for x in self.rows})==24 and all(x['passed']for x in self.rows)
  except BaseException as e:self.fail(e);raise
  if self.mode=='shadow':self.baseline_pass=True
  return self.save()
 def close(self):
  errors=[]
  for obj,key,had,old,new in reversed(self.bindings):
   if obj.__dict__.get(key)is not new:errors.append(key);continue
   if had:setattr(obj,key,old)
   else:delattr(obj,key)
  self.bindings=[];assert not errors;return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR);a=Adapter(output);setattr(model,ATTR,a)
 try:return a.install(model)
 except BaseException:a.close();delattr(model,ATTR);raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 try:return a.close()
 finally:delattr(model,ATTR)
