"""Native warmup; request selected helpers are typed TileIR original-address writes."""
from staged_ownership_runtime_v1.observer import OwnershipObserver as Original
from tileir_staged_typed_v2.contract import snapshot
from staged_typed_shadow_runtime_v2.evidence import check_stream,check_ledger
from .lifetime import Destination
from .oracle import expected
NAMES=('_staged_write_indices','_staged_write_starts','_staged_write_contents','_staged_write_cu_lens')
class OwnershipObserver(Original):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.rows=[];self.destinations=[]
 def report(self):
  r=super().report();r['replacement_rows']=self.rows;return r
 def release_after_sync(self):
  super().release_after_sync();self.destinations.clear()
 def _destination(self,owner):
  found=[d for d in self.destinations if d.owner is owner];assert len(found)==1
  return found[0].check()
 def _run(self,label,native,args,kwargs,pre=None,post=None):
  if self.phase()!='request'or label not in ('staged','fused'):
   ret=super()._run(label,native,args,kwargs,pre,post)
   if label=='staged_setup':
    try:self.destinations.append(Destination(args[0]))
    except BaseException:
     self.failure=dict(phase=self.phase(),label=label,stage='after_native_destination');raise RuntimeError('destination registration failed')from None
   return ret
  if self.failed:raise RuntimeError('inplace observer latched')
  row=dict(label=label,passed=False,native_returns=0,candidate_returns=0);self.rows.append(row)
  key='request.'+label
  try:
   assert not kwargs and len(self.rows)<=140
   assert all(getattr(cls,name)is wrapper for cls,name,original,wrapper in self.patches)
   owners=[args[0]]if label=='staged'else list(args[1]);assert 0<len(owners)<=4
   row['ownership']=[self._destination(o)for o in owners]
   data=[snapshot(o)for o in owners];row['typed_nonempty']=[bool(x[0])for x in data];row['nonempty']=any(row['typed_nonempty'])
   selected=[x for x in self.rows if x['label']==label]
   assert len(selected)<=(123 if label=='staged'else 17)
   assert sum(x.get('nonempty',False)for x in selected)<=(5 if label=='staged'else 1)
   assert sum(not x.get('nonempty',False)for x in selected)<=(118 if label=='staged'else 16)
   outputs=[o.gpu for o in owners]
   assert all(t.storage_offset()==0 and t.numel()*t.element_size()==t.untyped_storage().nbytes()<=67108864 and not t.requires_grad for t in outputs)
   if pre:pre(*args,**kwargs) # Original public table/owner guard, never original helper.
   listrefs=[[getattr(o,n)for n in NAMES]for o in owners]
   # Pool objects/cursors are retained but unused by candidate; policy is not changed.
   pools=[(o,tuple((name,getattr(o,name,None),getattr(getattr(o,name,None),'_curr',None))for name in ('write_indices','write_starts','write_cu_lens','write_contents')))for o in owners]
   if label=='fused':pools.append((args[0],tuple((n,getattr(args[0],n,None),getattr(getattr(args[0],n,None),'_curr',None))for n in ('group_ids','indices','starts','cu_lens'))))
   import torch
   from tileir_staged_typed_v2 import api
   def raw(t):return t.view(torch.uint8).cpu().numpy().tobytes()
   wanted=[expected(raw(t),tuple(t.shape),tuple(t.stride()),str(t.dtype),d)for t,d in zip(outputs,data)]
   ledger=[];row['launch_ledger']=ledger;stream=torch.cuda.current_stream()
   self.counts[key+'.calls']+=1;self.counts[key+'.candidate_attempts']+=1
   if label=='staged':ret=api.apply(outputs[0],*data[0],ledger=ledger)
   else:ret=api.fused(outputs,data,ledger=ledger)
   row['candidate_returns']=1;self.counts[key+'.candidate_returns']+=1;assert ret is None
   row['candidate_stream_unchanged']=check_stream(stream,torch.cuda.current_stream());check_ledger(row)
   torch.cuda.synchronize()
   assert all(raw(t)==w for t,w in zip(outputs,wanted));row['raw_equal']=True
   assert [snapshot(o)for o in owners]==data
   for o,records in pools:
    assert all(getattr(o,n,None)is obj and getattr(obj,'_curr',None)==cursor for n,obj,cursor in records)
   row['host_source_readonly']=True;row['native_metadata_pools_untouched']=True
   for o in owners:self._destination(o)
   if row['nonempty']:
    for o in owners:o.clear_staged_writes()
   after=[snapshot(o)for o in owners];assert after==([((),(),(),())for o in owners]if row['nonempty']else data)
   assert all(all(getattr(o,n)is ref for n,ref in zip(NAMES,refs))for o,refs in zip(owners,listrefs))
   if post:post(*args,**kwargs)
   check_stream(stream,torch.cuda.current_stream())
   row.update(native_list_semantics=True,original_addresses_preserved=True,output_bytes=sum(len(w)for w in wanted),passed=True)
   return None
  except BaseException:
   self.failure=dict(phase='request',label=label,stage='candidate_oracle_or_cleanup');raise RuntimeError('staged inplace qualification failed')from None
