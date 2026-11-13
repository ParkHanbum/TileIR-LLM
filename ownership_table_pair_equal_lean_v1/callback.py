from ownership_table_pair_equal_v1.api import equal
from ownership_table_pair_equal_v1.comparator import comparator as candidate_factory
from staged_ownership_runtime_v1.observer import comparator as original_factory

def request_comparator(torch,phase):
 state=dict(failed=False,tables=0,pair_attempts=0,pair_returns=0,uint64=0,int64=0)
 def pair(a,b):
  if state['failed'] or state['pair_attempts']>=68:raise RuntimeError('pair cap/latch')
  state['pair_attempts']+=1
  result=equal(a,b)
  if type(result)is not bool:raise TypeError('boolean result')
  dtype=str(a.dtype)
  if dtype not in ('torch.uint64','torch.int64'):raise ValueError('pair dtype')
  state[dtype.split('.')[-1]]+=1;state['pair_returns']+=1;return result
 candidate=candidate_factory(torch,pair);native=original_factory(torch)
 def callback(ptrs,strides,tensors):
  if state['failed']:raise RuntimeError('comparison latch')
  if phase()!='request':return native(ptrs,strides,tensors)
  try:
   if state['tables']>=34:raise RuntimeError('table cap')
   result=candidate(ptrs,strides,tensors);state['tables']+=1;return result
  except BaseException:state['failed']=True;raise
 def finish():
  if state!=dict(failed=False,tables=34,pair_attempts=68,pair_returns=68,uint64=34,int64=34):raise RuntimeError('pair census')
  return dict(table_checks=34,pair_checks=68,native_request_equal_calls=0,native_oracle_calls=0,raw_payload_copies=0,tileir_equal_calls=68,expected_table_construction='unchanged native H2D creation',host_boolean_boundary=True)
 return callback,finish
