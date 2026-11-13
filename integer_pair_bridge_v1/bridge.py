"""CPU-only lifecycle bridge draft. Numerical APIs and owners are unchanged."""
class Bridge:
 def __init__(self,cast,integer,lean):
  self.cast=cast;self.integer=integer;self.lean=lean;self.failed=None;self.restored=False;self.bound=False
 def latch(self,error):
  if self.failed is None:self.failed=repr(error)
  for state in (self.cast,self.integer):
   if state.failed is None:state.failed=self.failed
  controller=self.lean()
  if controller is not None:controller.latch('integer_pair',error)
 def bind_guards(self):
  assert not self.bound
  self.original=[]
  for state in (self.cast,self.integer):
   guard=state.guard;self.original.append((state,guard))
   def wrapped(_guard=guard):
    try:
     controller=self.lean()
     assert self.failed is None and self.cast.failed is None and self.integer.failed is None,'Integer pair failed'
     assert controller is None or controller.failed is None,'Other component failed'
     return _guard()
    except BaseException as e:self.latch(e);raise
   state.guard=wrapped
  self.bound=True
 def start(self,mode):
  assert self.bound and self.failed is None
  try:self.cast.start(mode);self.integer.start(mode)
  except BaseException as e:self.latch(e);raise
 def finish(self):
  assert self.failed is None
  try:return dict(embedding_cast=self.cast.finish(),sampler_integer=self.integer.finish())
  except BaseException as e:self.latch(e);raise
 def close(self,close_lean):
  # Installed cast -> integer -> lean. Always attempt all inverse restorations.
  results={};errors=[]
  for name,fn in [('lean',close_lean),('sampler_integer',self.integer.close),('embedding_cast',self.cast.close)]:
   try:
    result=fn();assert result.get('restored')is True;results[name]=result
   except BaseException as e:errors.append((name,repr(e)))
  self.restored=not errors
  if errors:raise RuntimeError('Integer pair independent restoration failed',errors)
  for state,guard in self.original:state.guard=guard
  return dict(restored=True,components=results)
 def require_release(self):
  assert self.restored,'Keep owners until every restoration succeeds'
