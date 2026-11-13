"""Draft typed-owned zero API, imports CUDA only after geometry validation."""
from zero_kv_phase2_typed_plan_v1.geometry import partition
from zero_kv_phase2_typed_plan_v1.owned_view import chunk_view
def zero_owned_intervals(allocation,intervals,*,torch,ct,ledger=None):
 if ledger is None:ledger=dict(attempted=0,returned=0)
 assert ledger==dict(attempted=0,returned=0)
 allocation.check();chunks=partition(allocation.size,intervals)
 targets=[(lo,hi)for kind,lo,hi in chunks if kind=='target']
 if not targets:return dict(target_bytes=0,launches=0)
 from zero_kv_phase2_typed_plan_v1.kernel import zero_words
 stream=torch.cuda.current_stream(allocation.owner.device)
 for lo,hi in targets:
  view=chunk_view(torch,allocation,lo,hi)
  ledger['attempted']+=1
  ct.launch(stream,((view.numel()+1023)//1024,),(zero_words),(view,))
  ledger['returned']+=1
 return dict(target_bytes=sum(hi-lo for lo,hi in targets),launches=ledger['returned'])
