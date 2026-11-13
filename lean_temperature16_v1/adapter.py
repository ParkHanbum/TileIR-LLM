"""Source-only pending actual temp16 endpoint; same checked API, no oracle."""
from temperature16_combined_qualification_v1.adapter import Greedy
class Checks:
 def __init__(self,controller):self.controller=controller;self.active=False;self.busy=False
 def start(self):
  assert not self.active and self.controller.failed is None
  self.calls=0;self.active=True
 def predicate(self,temp,idx):
  try:
   import torch
   from public_temperature_zero_v1.api import selected_all_zero
   assert self.active and self.controller.active and self.controller.failed is None and not self.busy and self.calls<16
   for x,dtype,size in ((temp,torch.float32,128),(idx,torch.int64,256)):
    assert tuple(x.shape)==(32,)and tuple(x.stride())==(1,)and x.storage_offset()==0 and x.untyped_storage().nbytes()==size
    assert x.dtype==dtype and x.device.type=='cuda'and not x.requires_grad and not x.is_neg()and not x.is_conj()
   assert temp.device==idx.device
   self.busy=True;stream=torch.cuda.current_stream(temp.device).cuda_stream
   out=selected_all_zero(temp,idx)
   assert type(out)is bool and torch.cuda.current_stream(temp.device).cuda_stream==stream
   self.calls+=1;return out
  except BaseException as e:self.controller.latch('temperature16-lean',e);raise
  finally:self.busy=False
 def finish(self):
  assert self.active and not self.busy and self.controller.failed is None and self.calls==16
  self.active=False
  return dict(calls=16,native_predicates=0,tile_predicates=16,extra_checked_range_calls=16,temperature_original_retained=0,temperature_followup_hook=True,diagnostic_native_oracle=False,diagnostic_raw_compare=False)

def annotate_sampling_receipt(sampling96,temperature16):
 """Call only after BOTH counters finish. Mutates only the new result dict."""
 assert temperature16['calls']==16 and temperature16['native_predicates']==0
 assert sampling96['calls']==96 and sampling96['temperature_original_retained']==16
 sampling96['temperature_original_retained']=0
 sampling96['temperature_followup_hook']=True
 sampling96['temperature_observation_slots']=16
 return sampling96
