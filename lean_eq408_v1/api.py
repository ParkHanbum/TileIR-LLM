"""Lean EQ408 checks: unchanged public numerical API, no native oracle/readback diagnostics."""
from collections import Counter
LIMITS={'attention.q':128,'attention.k':128,'attention.v':128,'gdn.initial':24}
MAP={'q_descale':'attention.q','k_descale':'attention.k','v_descale':'attention.v'}
class Checks:
 def __init__(self,controller,candidate=None):
  self.controller=controller;self.candidate=candidate;self.active=False;self.counts=Counter();self.returns=Counter();self.busy=False;self.failed=None
 def start(self):
  assert not self.active and self.failed is None and self.controller.failed is None
  self.counts=Counter();self.returns=Counter();self.active=True
 def predicate(self,site,x,value):
  try:
   assert self.active and self.controller.active and self.controller.failed is None and self.failed is None and not self.busy
   assert site in LIMITS and self.counts[site]<LIMITS[site] and type(value)is int and value==(0 if site=='gdn.initial'else 1)
   if site=='gdn.initial':assert tuple(x.shape)==(32,32,128,128) and tuple(x.stride())==(524288,16384,128,1) and x.storage_offset()==0
   else:assert tuple(x.shape)==(32,4) and tuple(x.stride())==(0,0)
   assert str(x.dtype)=='torch.float32'
   self.busy=True;self.counts[site]+=1
   candidate=self.candidate
   if candidate is None:
    from public_eq_all_guard_v2.api import eq_all
    candidate=eq_all
   value=candidate(x,value);assert type(value)is bool
   self.returns[site]+=1
   return value  # False remains the original assert's decision, not converted to success.
  except BaseException as e:
   self.failed=type(e).__name__;self.controller.latch('eq408',e);raise
  finally:self.busy=False
 def equal_one(self,name,x):return self.predicate(MAP[name],x,1)
 def initial_zero(self,x,value):return self.predicate('gdn.initial',x,value)
 def finish(self):
  assert self.active and not self.busy and self.failed is None and self.controller.failed is None and dict(self.counts)==dict(self.returns)==LIMITS
  self.active=False
  return dict(calls=408,counts=dict(self.counts),returns=dict(self.returns),native_predicates=0,tile_predicates=408,failed=None)
