from contextlib import contextmanager
@contextmanager
def launch_spy(ct,targets,guards,events=None):
 original=ct.launch;counts=dict(target=0,guard=0,other=0)
 def wrapped(stream,grid,kernel,args,*a,**kw):
  key='target' if any(kernel is x for x in targets)else'guard' if any(kernel is x for x in guards)else'other'
  counts[key]+=1
  if events is not None:events.append(dict(kind=key,grid=tuple(grid),args=args))
  return original(stream,grid,kernel,args,*a,**kw)
 ct.launch=wrapped
 try:yield counts
 finally:
  assert ct.launch is wrapped
  ct.launch=original
