"""Bounded verification; plans/offsets/digests stay process-local, aggregate receipt only."""
from zero_kv_phase2_typed_plan_v1.geometry import from_registry,CHUNK
from zero_kv_phase2_typed_plan_v1.owned_view import chunk_view
from .registry import require
IO_CAP=16*1024**3

def validate_scope(registry,ids,target):
 require(len(registry.segments)==8 and len(registry.allocations)==1,'actual_geometry_scope')
 require(len(ids)==32 and len(set(ids))==32 and target==553648128,'actual_geometry_scope')

def verify_and_zero(registry,ids,*,read_chunk,zero,synchronize,record):
 registry.check();plans=from_registry(registry,ids)
 total=sum(a.size for a,p in plans);target=sum(hi-lo for a,p in plans for k,lo,hi in p if k=='target');io=2*total-target
 validate_scope(registry,ids,target)
 require(io<=IO_CAP,'proof_io_budget');require(target>0,'empty_selected_union')
 record.update(phase='before',allocation_bytes=total,target_bytes=target,planned_d2h_bytes=io,cpu_chunk_limit=CHUNK,outside_hash_preserved=False,target_all_zero=False,zero_attempted=0,zero_returned=0)
 synchronize();before=[]
 for a,parts in plans:
  for kind,lo,hi in parts:
   if kind=='outside':before.append(read_chunk(a,lo,hi,False))
 registry.check();record['phase']='zero'
 for a,parts in plans:
  intervals=[(lo,hi)for kind,lo,hi in parts if kind=='target']
  ledger=dict(attempted=0,returned=0)
  try:zero(a,intervals,ledger)
  finally:
   record['zero_attempted']+=ledger['attempted'];record['zero_returned']+=ledger['returned']
 synchronize();registry.check();record['phase']='after';i=0
 for a,parts in plans:
  for kind,lo,hi in parts:
   if kind=='outside':
    require(read_chunk(a,lo,hi,False)==before[i],'outside_hash_mismatch');i+=1
   else:require(read_chunk(a,lo,hi,True)is True,'target_nonzero')
 expected=sum(k=='target'for a,p in plans for k,lo,hi in p)
 require(record['zero_attempted']==record['zero_returned']==expected,'candidate_call_count')
 record.update(phase='passed',outside_hash_preserved=True,target_all_zero=True,expected_source_launches=expected,source_calls_not_physical=True)
 return None

def run_proof(registry,ids,record):
 import hashlib,torch,cuda.tile as ct
 from zero_kv_typed_synthetic_v2.api import zero_owned_intervals
 stream=torch.cuda.current_stream(registry.strong_owners[0].device)
 def read(a,lo,hi,target):
  view=chunk_view(torch,a,lo,hi);cpu=view.cpu()
  if target:return int(torch.count_nonzero(cpu).item())==0
  return hashlib.sha256(memoryview(cpu.numpy()).cast('B')).digest()
 def zero(a,intervals,ledger):return zero_owned_intervals(a,intervals,torch=torch,ct=ct,ledger=ledger)
 verify_and_zero(registry,ids,read_chunk=read,zero=zero,synchronize=stream.synchronize,record=record)
 require(torch.cuda.current_stream(registry.strong_owners[0].device).cuda_stream==stream.cuda_stream,'stream_changed')
