"""CPU controller helpers; current-stream candidate APIs remain responsible for validation."""
from collections import Counter
import functools
class Base:
 def __init__(self,controller,key,expected):
  self.controller=controller;self.key=key;self.expected=expected;self.bindings=[];self.counts=Counter();self.pending=None
 def guard(self):
  assert self.controller.active and self.controller.failed is None,'Lean call outside active qualified request or after failure'
  assert sum(self.counts.values())<self.expected,'Lean supported census exceeded'
 def call(self,owner,phase,fn,*args,**kwargs):
  self.guard()
  try:
   result=fn(*args,**kwargs);self.counts[(owner,phase)]+=1;return result
  except BaseException as e:self.controller.latch(self.key,e);raise
 def bind(self,obj,name,new):
  old=getattr(obj,name);had=name in vars(obj);prior=vars(obj).get(name)
  @functools.wraps(new)
  def guarded(*args,**kwargs):
   try:
    self.guard();return new(*args,**kwargs)
   except BaseException as e:self.controller.latch(self.key,e);raise
  self.bindings.append((obj,name,old,had,prior,guarded));setattr(obj,name,guarded);return old
 def start(self):self.counts.clear();assert self.pending is None
 def finish(self):
  assert self.pending is None and sum(self.counts.values())==self.expected,(self.key,dict(self.counts))
  return dict(calls=self.expected,owners=[dict(owner=k[0],phase=k[1],calls=v)for k,v in sorted(self.counts.items())],selected_native_calls=0,selected_fallback_calls=0)
 def close(self):
  errors=[]
  for obj,name,old,had,prior,new in reversed(self.bindings):
   try:
    assert getattr(obj,name)is new,(self.key,name,'foreign binding')
    if had:setattr(obj,name,prior)
    else:delattr(obj,name)
    assert getattr(obj,name)==old,(self.key,name,'restore mismatch')
   except BaseException as e:errors.append(repr(e))
  self.bindings=[];assert not errors,errors;return dict(restored=True)

def phase_index(index):
 assert type(index)is int and 0<=index<16
 return 'prefill'if index==0 else'decode'
def call_phase(counter,owner,rows):
 n=sum(v for(k,p),v in counter.items()if k==owner);phase=phase_index(n)
 assert rows==(1696 if phase=='prefill'else 32),(owner,phase,rows)
 return phase
