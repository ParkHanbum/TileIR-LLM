"""Explicit observer-instance callback replacement, never a global Torch patch."""
import inspect,hashlib
from pathlib import Path
from staged_ownership_runtime_v1.observer import comparator as original_factory
from .qualifier import request_comparator
class Binding:
 def __init__(self,observer,torch,phase,record=lambda value:None):
  self.observer=observer;self.old=observer.compare;self.installed=False;self.failed=False
  expected=original_factory(torch)
  if self.old.__code__ is not expected.__code__:raise ValueError('actual comparator code origin')
  closure=inspect.getclosurevars(self.old).nonlocals
  if set(closure)!= {'torch'} or closure['torch'] is not torch:raise ValueError('actual comparator closure')
  path=inspect.getsourcefile(original_factory)
  self.origin=dict(module=self.old.__module__,qualname=self.old.__qualname__,source_path=path,source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),firstlineno=self.old.__code__.co_firstlineno,code_identity_verified=True,torch_closure_identity_verified=True)
  callback,self.complete=request_comparator(torch,phase,record)
  def guarded(*a,**k):
   if self.failed or observer.compare is not guarded:raise RuntimeError('latched or changed compare binding')
   try:return callback(*a,**k)
   except BaseException:self.failed=True;raise
  self.new=guarded
 def install(self):
  if self.installed or self.observer.compare is not self.old:raise RuntimeError('compare install collision')
  self.observer.compare=self.new;self.installed=True
 def finish(self):
  if self.failed or not self.installed or self.observer.compare is not self.new:raise RuntimeError('compare finish identity')
  result=self.complete();result['actual_callback_origin']=self.origin;return result
 def close(self):
  if self.installed:
   if self.observer.compare is not self.new:self.failed=True;raise RuntimeError('foreign compare binding; retained owner')
   self.observer.compare=self.old;self.installed=False
  return dict(restored=True,original_callback_identity=self.observer.compare is self.old)
