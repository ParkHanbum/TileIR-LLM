"""Lean callbacks only; unchanged exact owned range/sampling bodies and APIs.
No native predicates, snapshots, byte comparisons, temperature observation logs.
Scope metadata checks, typed counts and shared first-failure latch remain.
"""
from collections import Counter
from sampling_guards144_runtime_v2 import adapter as proof
from sampling_guards144_runtime_v2.checks import LIMITS
from sampling_guards144_runtime_v2.range_adapter import LIMITS as RANGE_LIMITS
ATTR=proof.ATTR
class Range:
 def __init__(self,c):self.controller=c;self.active=False
 def start(self,mode):
  assert mode=='replace'and not self.active and self.controller.failed is None
  self.counts=Counter();self.active=True
 def predicate(self,site,x,lower,upper):
  try:
   import torch
   from public_i64_range_guard_v1.api import all_in_range_i64
   assert self.active and self.controller.active and self.controller.failed is None
   assert site in RANGE_LIMITS and self.counts[site]<16 and lower==0
   assert tuple(x.shape)==((32,6)if site=='scores'else(32,))and x.dtype==torch.int64
   assert type(upper)is int and(upper>0 if site=='greedy'else upper==248320)
   assert x.device.type=='cuda'and not x.requires_grad and x.untyped_storage().nbytes()<=1048576
   out=all_in_range_i64(x,lower,upper);assert type(out)is bool
   self.counts[site]+=1;return out
  except BaseException as e:self.controller.latch('range48-lean',e);raise
 def finish(self):
  assert self.active and self.controller.failed is None and dict(self.counts)==RANGE_LIMITS
  self.active=False;return dict(calls=48,counts=dict(self.counts),native_predicates=0,tile_predicates=48)
class Sampling:
 def __init__(self,c):self.controller=c;self.active=False
 def start(self,mode):
  assert mode=='replace'and not self.active and self.controller.failed is None
  self.counts=Counter();self.temperature_count=0;self.active=True
 def enter(self,site,x):
  import torch
  assert self.active and self.controller.active and self.controller.failed is None and site in LIMITS and self.counts[site]<16
  assert tuple(x.shape)==(32,248320)and tuple(x.stride())==(248320,1)and x.storage_offset()==0
  assert x.dtype==(torch.float32 if site.startswith('greedy')else torch.bfloat16)
  assert x.is_cuda and not x.requires_grad and not x.is_neg()and not x.is_conj()
 def predicate(self,site,x,mode):
  try:
   from public_sampling_guard_v1.api import classify
   self.enter(site,x)
   assert mode=={'greedy.nan':'has_nan','greedy.posinf':'has_posinf','greedy.rows':'rows_have_finite','ranks.nan':'has_nan','ranks.posinf':'has_posinf'}[site]
   out=classify(x,mode);assert type(out)is bool;self.counts[site]+=1;return out
  except BaseException as e:self.controller.latch('sampling96-lean',e);raise
 def gather(self,x,ids):
  try:
   import torch
   from public_sampling_guard_v1.api import selected_columns_checked
   self.enter('ranks.gather',x);assert ids.dtype==torch.int64 and tuple(ids.shape)==(32,)and ids.device==x.device
   out=selected_columns_checked(x,ids)
   assert tuple(out.shape)==(32,1)and out.dtype==x.dtype and out.device==x.device and out.is_contiguous()
   assert out.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()
   self.counts['ranks.gather']+=1;return out
  except BaseException as e:self.controller.latch('sampling96-lean',e);raise
 def observe_temperature(self,temp,idx):
  # No payload/metadata recording: retain original temperature predicate upstream.
  assert self.active and self.controller.active and self.controller.failed is None and self.temperature_count<16
  self.temperature_count+=1
 def finish(self):
  assert self.active and self.controller.failed is None and dict(self.counts)==LIMITS and self.temperature_count==16
  self.active=False
  return dict(calls=96,counts=dict(self.counts),original_predicates=0,original_gathers=0,tile_predicates=80,tile_gathers=16,additional_internal_range_predicates=16,temperature_original_retained=16)
class Controller(proof.Controller):
 def __init__(self,model):
  super().__init__(model);self.range48=Range(self);self.sampling96=Sampling(self)
 def start_mode(self,mode='replace'):return super().start_mode(mode)
def install(model):
 assert not hasattr(model,ATTR);c=Controller(model);setattr(model,ATTR,c);return c.install()
def start(model):return getattr(model,ATTR).start_mode()
def finish(model):return getattr(model,ATTR).finish()
def close(model):return proof.close(model)
def release(model):return proof.release(model)
