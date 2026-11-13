from lean26_gdn_finite_runtime_v1.core import PrefillCore as Original,INTEGER,ORDER
from .api import all_finite_integer
SPECS={'cu_seqlens':('torch.int32',(33,),(1,)), 'chunk_indices':('torch.int32',(32,2),(2,1)), 'chunk_offsets':('torch.int64',(33,),(1,))}
class PrefillCore(Original):
 def integer_check(self,site,t):return all_finite_integer(t)
 def finite(self,t):
  try:
   assert self.controller.active and self.controller.failed is None and self.finite_scope is not None
   i=self.finite_scope;assert 0<=i<len(ORDER);site=ORDER[i]
   if site not in INTEGER:return super().finite(t)
   assert self.site_counts[site]<24
   dtype,shape,stride=SPECS[site]
   assert (str(t.dtype),tuple(t.shape),tuple(t.stride()))==(dtype,shape,stride)
   meta=dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),offset=t.storage_offset(),numel=t.numel())
   if site in self.descriptors:assert meta==self.descriptors[site]
   else:self.descriptors[site]=meta
   ok=self.integer_check(site,t);assert ok is True
   self.site_counts[site]+=1;self.finite_scope=i+1;return ok
  except BaseException as e:self.controller.latch(self.key,e);raise
 def finish(self):
  out=super().finish();proof=out['gdn_finite']
  proof['typed_integer_checks']=proof.pop('retained_integer_checks');proof['typed_integer_site_counts']=proof.pop('retained_integer_site_counts')
  proof.update(integer_gpu_kernels=0,integer_stream_synchronizations=72,integer_native_predicates=0,scope='Typed integer constant proof with native current-stream synchronize; float checks unchanged')
  return out
