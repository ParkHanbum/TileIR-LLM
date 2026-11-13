"""Predicate-only controller mixin; exact model proof checked once before use."""
from collections import Counter
from finite_504_runtime_v1.adapter import LIMITS,Post,KV,Attention
from .proof import verify
class Checks:
 def __init__(self,controller):
  self.proof=verify();self.controller=controller;self.active=False;self.counts=Counter();self.busy=False
 def start(self):
  assert not self.active and self.controller.failed is None
  self.counts=Counter();self.active=True
 def finite(self,site,x):
  from tileir_validation_primitives_v3.api import all_finite
  from tileir_kv_strided_finite_v1.api import all_finite_value
  try:
   assert self.active and self.controller.active and self.controller.failed is None and not self.busy
   assert site in LIMITS and self.counts[site]<LIMITS[site]
   self.busy=True
   # Retained domain bound. No full-owner snapshots or native predicate oracle.
   assert x.untyped_storage().nbytes()<=128*1024*1024
   value=all_finite_value(x)if site=='kv.value'else all_finite(x)
   assert type(value)is bool
   self.counts[site]+=1
   return value # Original assertion/short-circuit owns the false failure.
  except BaseException as e:self.controller.latch('finite504',e);raise
  finally:self.busy=False
 def finish(self):
  assert self.active and self.controller.failed is None and dict(self.counts)==LIMITS
  self.active=False
  return dict(calls=504,dense=376,strided=128,counts=dict(self.counts),native_predicates=0,tile_predicates=504,diagnostic_owner_copies=0,scope='Predicate dispatch only; historical raw/read-only proof separate')

def replace_specs(specs):
 replacements={'post_conv':Post,'attention_kv':KV,'attention_core':Attention}
 assert all(sum(k==target for k,c,n in specs)==1 for target in replacements)
 return tuple((k,replacements.get(k,c),n)for k,c,n in specs)
