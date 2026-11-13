"""CPU-only shared failure bridge. No numerical APIs, runtime imports or engine."""
class Bus:
 def __init__(self,base=lambda:None):self.base=base;self.failure=None;self.closers=[];self.restored=False;self.original_methods=[]
 @property
 def failed(self):
  c=self.base();return self.failure or (getattr(c,'failed',None)if c is not None else None)
 @property
 def active(self):
  c=self.base();return c is not None and c.active
 def latch(self,key,error):
  if self.failure is None:self.failure=dict(component=key,error_type=type(error).__name__)
  c=self.base()
  if c is not None and c.failed is None:c.latch(key,RuntimeError('Shared component failure'))
 def guard(self):
  if self.failed is not None:raise RuntimeError('Shared first failure latched')
 def entry(self,label,fn):
  def guarded(*a,**kw):
   self.guard()
   try:return fn(*a,**kw)
   except BaseException as e:self.latch(label,e);raise
  return guarded
 def bind_scalar(self,state):
  # Must happen after ctor but BEFORE install: Binding retains replacement object.
  state.check_global=self.guard
  for name in ('call','finish','start'):
   old=getattr(state,name);self.original_methods.append((state,name,old));setattr(state,name,self.entry('scalar/'+name,old))
  old=state.binding.replacement;state.binding.replacement=self.entry('scalar/owned_add_request',old)
  self.original_methods.append((state.binding,'replacement',old))
 def bind_metadata(self,state):
  # Must happen BEFORE state.install constructs owned source callbacks.
  for name in ('subq','subs','gt','full','invoke','start','finish'):
   old=getattr(state,name);self.original_methods.append((state,name,old));setattr(state,name,self.entry('metadata/'+name,old))
  old=state.guard
  def guard(fn):return self.entry('metadata/owned_entry',old(fn))
  state.guard=guard;self.original_methods.append((state,'guard',old))
 def install(self,label,install,close):
  self.guard();self.closers.append((label,close)) # includes partially installed controller
  try:return install()
  except BaseException as e:self.latch('install/'+label,e);raise
 def close(self):
  errors=[];order=[]
  for label,close in reversed(self.closers):
   order.append(label)
   try:close()
   except BaseException as e:errors.append(dict(component=label,error_type=type(e).__name__))
  self.restored=not errors
  if errors:raise RuntimeError('Independent reverse restoration failed',errors)
  for obj,name,old in reversed(self.original_methods):setattr(obj,name,old)
  return dict(restored=True,order=order)
 def require_release(self,*,synchronized,shutdown):
  if not(self.restored and synchronized and shutdown):raise RuntimeError('Retain owners until cleanup complete')
