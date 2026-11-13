from copy import deepcopy
from public_post_update_guard_v1.api import Guard
from public_post_update_guard_v1.binding import Binding
class Checks:
 def __init__(self,owner):self.owner=owner;self.failed=False;self.mode=None;self.rows=[];self.shadow_pass=False;self.pending=[];self.binding=Binding(owner,self.validate)
 def start(self,mode):
  assert not self.failed and mode in('shadow','replace')and (mode!='replace'or self.shadow_pass)
  assert not self.owner.failed and not self.owner.active
  self.mode=mode;self.rows=[]
 def validate(self,torch,args):
  if self.failed:raise RuntimeError('Post guard failure latched')
  try:
   assert self.mode in('shadow','replace')and len(self.rows)<16
   assert self.owner.call is self.binding.copied,'Owned guard method changed'
   assert not self.owner.failed and self.owner.active and self.owner.controller is not None and self.owner.controller.failed is None and self.owner.controller.active
   def meta():return [(id(x),x.data_ptr(),x.untyped_storage().data_ptr(),x.untyped_storage().nbytes(),x.storage_offset(),tuple(x.shape),tuple(x.stride()),str(x.dtype))for x in args]
   # Existing exact metadata is checked by copied caller before this callback.
   from tileir_vllm_post_update_v3.contract import validate_metadata,validate_values
   validate_metadata(args);original=meta();stream=torch.cuda.current_stream(args[0].device).cuda_stream
   total=sum(x.untyped_storage().nbytes()for x in args);assert total<=64*1024*1024
   def snapshot():return tuple(x.detach().view(-1).view(torch.uint8).cpu().numpy().tobytes()for x in args)
   before=snapshot();row=dict(mode=self.mode,index=len(self.rows)+1,native_calls=0,candidate_calls=0,passed=False,owner_bytes=total,original_metadata=original);self.rows.append(row)
   native_error=None
   if self.mode=='shadow':
    row['native_calls']=1
    try:validate_values(torch,args)
    except ValueError as e:native_error=str(e)
    assert snapshot()==before,'Native guard mutated backing'
   guard=Guard();self.pending.append(guard);row['candidate_calls']=1;candidate_error=None
   try:guard.validate(torch,args)
   except ValueError as e:candidate_error=str(e)
   assert snapshot()==before and meta()==original and torch.cuda.current_stream(args[0].device).cuda_stream==stream
   row.update(native_error=native_error,candidate_error=candidate_error,readonly=True,metadata_preserved=True,same_stream=True,stages=list(guard.stages))
   assert not guard.pending;self.pending.pop()
   if self.mode=='shadow':assert native_error==candidate_error,'Same-input error mismatch'
   row['passed']=True
   if candidate_error is not None:raise ValueError(candidate_error)
   return None
  except BaseException as e:
   self.failed=True;self.owner.latch('post_update_guard',e);raise
 def finish(self):
  assert not self.failed and len(self.rows)==16 and not self.pending
  assert all(r['passed']and r['candidate_error']is None and r['stages']==list(range(9))for r in self.rows)
  if self.mode=='shadow':self.shadow_pass=True
  return deepcopy(dict(mode=self.mode,rows=self.rows,calls=16,shadow_pass=self.shadow_pass,failed=self.failed))
