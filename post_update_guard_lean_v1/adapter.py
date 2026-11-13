"""Lean owned predicate binding. Numerical API remains unchanged."""
from public_post_update_guard_v1.api import Guard
from post_update_guard_runtime_v2.binding import Binding
class Checks:
 def __init__(self,owner):
  self.owner=owner;self.binding=Binding(owner,self.validate);self.failed=False;self.active=False;self.calls=0;self.stages=0;self.pending=[]
 def start(self):
  if self.failed or self.active or self.pending:raise RuntimeError('Post-update lean lifecycle invalid')
  self.calls=0;self.stages=0;self.active=True
 def validate(self,torch,args):
  if self.failed:raise RuntimeError('Post guard failure latched')
  try:
   o=self.owner
   if not self.active or self.calls>=16:raise RuntimeError('Post-update lean call scope')
   if vars(o).get('call') is not self.binding.copied:raise RuntimeError('Owned guard method changed')
   if o.failed or not o.active or o.controller is None or o.controller.failed is not None or not o.controller.active:raise RuntimeError('Post-update owner inactive or failed')
   # Original copied call retains metadata/preflight checks. Guard repeats exact
   # metadata, typed views, nine short-circuit stages and current-stream checks.
   guard=Guard();self.pending.append(guard)
   guard.validate(torch,args)
   if guard.pending or guard.failed or guard.stages!=list(range(9)):raise RuntimeError('Incomplete post-update predicate')
   self.pending.pop();self.calls+=1;self.stages+=len(guard.stages)
  except BaseException as e:
   self.failed=True;self.owner.latch('post_update_guard',e);raise
 def finish(self):
  if self.failed or not self.active or self.calls!=16 or self.stages!=144 or self.pending:raise RuntimeError('Incomplete post-update lean request')
  self.active=False
  return dict(calls=self.calls,stages=self.stages,native_predicate_calls=0,backing_snapshots=0,failed=False)
