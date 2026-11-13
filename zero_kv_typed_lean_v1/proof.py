"""Lean launch-only path: no owner-payload reads/hashes or correctness oracle."""
from zero_kv_phase2_typed_plan_v1.geometry import from_registry,CHUNK
from zero_kv_typed_runtime_v2.proof import validate_scope,IO_CAP
from .registry import require

def launch_owned(registry,ids,*,zero,synchronize,record):
 registry.check();plans=from_registry(registry,ids)
 total=sum(a.size for a,p in plans);target=sum(hi-lo for a,p in plans for k,lo,hi in p if k=='target')
 validate_scope(registry,ids,target)
 require(2*total-target<=IO_CAP,'proof_io_budget') # Retains proven allocation budget domain, no payload transfers.
 record.update(phase='launch',allocation_bytes=total,target_bytes=target,owner_payload_read_bytes=0,owner_hashes_computed=0,target_oracle_executed=False,outside_oracle_executed=False,zero_attempted=0,zero_returned=0)
 synchronize();registry.check()
 for a,parts in plans:
  intervals=[(lo,hi)for kind,lo,hi in parts if kind=='target'];ledger=dict(attempted=0,returned=0)
  try:zero(a,intervals,ledger)
  finally:
   record['zero_attempted']+=ledger['attempted'];record['zero_returned']+=ledger['returned']
 synchronize();registry.check();expected=sum(k=='target'for a,p in plans for k,lo,hi in p)
 require(record['zero_attempted']==record['zero_returned']==expected,'candidate_call_count')
 record.update(phase='passed',expected_source_launches=expected,source_calls_not_physical=True)

def run_proof(registry,ids,record):
 import torch,cuda.tile as ct
 from zero_kv_typed_synthetic_v2.api import zero_owned_intervals
 stream=torch.cuda.current_stream(registry.strong_owners[0].device)
 def zero(a,intervals,ledger):return zero_owned_intervals(a,intervals,torch=torch,ct=ct,ledger=ledger)
 launch_owned(registry,ids,zero=zero,synchronize=stream.synchronize,record=record)
 require(torch.cuda.current_stream(registry.strong_owners[0].device).cuda_stream==stream.cuda_stream,'stream_changed')
