"""Same-input clone qualifier only; actual production returns always native."""
from staged_ownership_runtime_v1.observer import OwnershipObserver as Original
from tileir_staged_typed_v2.contract import snapshot
class OwnershipObserver(Original):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.shadow_rows=[]
 def report(self):
  r=super().report();r['shadow_rows']=self.shadow_rows;return r
 def _run(self,label,native,args,kwargs,pre=None,post=None):
  if self.phase()!='request'or label not in ('staged','fused'):return super()._run(label,native,args,kwargs,pre,post)
  if self.failed:raise RuntimeError('ownership observer latched')
  row=dict(label=label,passed=False,native_returns=0,candidate_returns=0)
  self.shadow_rows.append(row)
  try:
   assert not kwargs,'Observed public positional call scope'
   assert len(self.shadow_rows)<=140
   owners=[args[0]]if label=='staged'else list(args[1])
   assert owners and len(owners)<=4
   for owner in owners:self._view(owner)
   data=[snapshot(o)for o in owners]
   row['nonempty']=any(bool(x[0])for x in data)
   original=[o.gpu for o in owners]
   assert all(t.storage_offset()==0 and t.numel()*t.element_size()==t.untyped_storage().nbytes()<=67108864 and not t.requires_grad for t in original)
   import torch
   from tileir_staged_typed_v2 import api
   clones=[t.clone()for t in original]
   result=super()._run(label,native,args,kwargs,pre,post)
   row['native_returns']=1;assert result is None
   # Native clears every group's staging lists if any group had writes.
   after=[snapshot(o)for o in owners]
   expected=[((),(),(),())for _ in owners]if row['nonempty']else data
   assert after==expected;row['native_list_semantics']=True
   ledger=[];row['launch_ledger']=ledger;row['typed_nonempty']=[bool(x[0])for x in data]
   candidate_stream=torch.cuda.current_stream()
   if label=='staged':ret=api.apply(clones[0],*data[0],ledger=ledger)
   else:ret=api.fused(clones,data,ledger=ledger)
   row['candidate_returns']=1;assert ret is None
   from .evidence import check_stream,check_ledger
   row['candidate_stream_unchanged']=check_stream(candidate_stream,torch.cuda.current_stream())
   check_ledger(row)
   torch.cuda.synchronize()
   assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8))for a,b in zip(original,clones))
   row.update(raw_equal=True,output_bytes=sum(t.numel()*t.element_size()for t in original),passed=True)
   return result
  except BaseException:
   if self.failure is None:self.failure=dict(phase='request',label=label,stage='shadow_after_native'if row['native_returns']else 'shadow_pre_or_uncertain')
   raise RuntimeError('staged clone qualifier failed')from None
