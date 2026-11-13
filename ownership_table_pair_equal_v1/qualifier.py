from .api import equal,validate,descriptor
class Qualifier:
 def __init__(self,torch,cap=68,record=lambda value:None):self.torch=torch;self.cap=cap;self.rows=[];self.failed=False;self.record=record;self.observed=[]
 def raw(self,t):return t.detach().contiguous().view(self.torch.uint8).cpu().numpy().tobytes()
 def pair(self,a,b):
  if self.failed or len(self.rows)>=self.cap:self.failed=True;raise RuntimeError('latched/cap')
  try:
   # Record non-address metadata before any candidate support rejection.
   describe=lambda t:dict(dtype=str(t.dtype),device=str(t.device),shape=list(t.shape),stride=list(t.stride()),offset=t.storage_offset(),storage_bytes=t.untyped_storage().nbytes(),requires_grad=t.requires_grad,is_neg=t.is_neg(),is_conj=t.is_conj(),layout=str(t.layout))
   self.observed.append(dict(index=len(self.rows),a=describe(a),b=describe(b)))
   self.record(dict(observed=self.observed,rows=self.rows,failed=self.failed))
   n=validate(a,b);stream=self.torch.cuda.current_stream(a.device).cuda_stream;meta=descriptor(a),descriptor(b);before=self.raw(a),self.raw(b)
   native=bool(self.torch.equal(a,b));candidate=equal(a,b)
   if native!=candidate or (self.raw(a),self.raw(b))!=before or (descriptor(a),descriptor(b))!=meta or self.torch.cuda.current_stream(a.device).cuda_stream!=stream:raise ValueError('pair raw/readonly mismatch')
   # Do not serialize pointer payloads or addresses; only shape/dtype/boolean.
   self.rows.append(dict(dtype=str(a.dtype),shape=[n],stride=[1],native=native,candidate=candidate,whole_inputs_readonly=True));return candidate
  except BaseException:self.failed=True;self.record(dict(observed=self.observed,rows=self.rows,failed=True));raise

def request_comparator(torch,phase,record=lambda value:None):
 """Explicit compare_entries dependency; never patches torch or resolves pointers."""
 from staged_ownership_runtime_v1.observer import comparator as original
 from .comparator import comparator as candidate
 q=Qualifier(torch,cap=68,record=record);native=original(torch);checked=candidate(torch,q.pair)
 q.table_checks=0
 def compare(ptrs,strides,tensors):
  if q.failed:raise RuntimeError('ownership comparison latched')
  if phase()!='request':return native(ptrs,strides,tensors)
  try:
   if q.table_checks>=34:raise RuntimeError('request table cap')
   with torch.profiler.record_function('ownership_table_pair_equal::same_input'):
    result=checked(ptrs,strides,tensors)
   q.table_checks+=1;return result
  except BaseException:q.failed=True;raise
 def finish():
  if q.failed or q.table_checks!=34 or len(q.rows)!=68:raise RuntimeError('request equality census')
  if sum(r['dtype']=='torch.uint64'for r in q.rows)!=34 or sum(r['dtype']=='torch.int64'for r in q.rows)!=34:raise ValueError('dtype census')
  if not all(r['native'] is True and r['candidate'] is True and r['whole_inputs_readonly']for r in q.rows):raise ValueError('expected valid runtime table')
  return dict(table_checks=34,pair_checks=68,native_diagnostic_oracles=68,rows=q.rows,actual_metadata=q.observed,pointer_values_serialized=False)
 return compare,finish
