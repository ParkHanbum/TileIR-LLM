"""Scoped public IrOp instance dispatch observation; original called exactly once."""
from threading import get_ident
class DispatchObserver:
 def __init__(self,ops,describe):
  self.ops=ops;self.describe=describe;self.saved=[];self.active=None;self.events={};self.owner=get_ident()
 def install(self):
  assert not self.saved
  try:
   for name,op in self.ops.items():
    had='dispatch' in op.__dict__;prior=op.__dict__.get('dispatch');original=op.dispatch
    def wrapper(*args,_name=name,_original=original,**kwargs):
     impl=_original(*args,**kwargs)
     if self.active is not None and get_ident()==self.owner:
      self.events.setdefault(self.active,[]).append(dict(op=_name,provider=impl.provider,inplace=impl.inplace,supported=impl.supported,implementation=self.describe(impl.impl_fn),functional_wrapper=self.describe(impl.func_impl_fn)))
     return impl
    self.saved.append((op,had,prior,wrapper));op.dispatch=wrapper
   return self
  except BaseException:self.close();raise
 def enter(self,key):
  assert self.active is None and get_ident()==self.owner
  self.active=key;self.events[key]=[]
 def leave(self,key):
  assert self.active==key;self.active=None
  return list(self.events[key])
 def close(self):
  self.active=None;conflicts=[]
  for op,had,prior,wrapper in reversed(self.saved):
   if op.__dict__.get('dispatch') is not wrapper:conflicts.append(type(op).__name__);continue
   if had:op.dispatch=prior
   else:delattr(op,'dispatch')
  self.saved=[]
  if conflicts:raise RuntimeError('Foreign IR dispatch bindings preserved: '+repr(conflicts))
