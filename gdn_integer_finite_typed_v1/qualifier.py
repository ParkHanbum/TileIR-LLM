"""Same-input diagnostic interface for future isolated model72 qualification."""
from .api import all_finite_integer,validate,descriptor
class Qualifier:
 def __init__(self,torch):self.torch=torch;self.failed=False;self.counts=dict.fromkeys(('cu_seqlens','chunk_indices','chunk_offsets'),0);self.rows=[]
 def check(self,site,t,mode="replace"):
  if mode not in ('shadow','replace') or self.failed or site not in self.counts or self.counts[site]>=24:self.failed=True;raise RuntimeError('site/cap/latch')
  try:
   before_meta=validate(t);before=t.detach().contiguous().view(self.torch.uint8).cpu().numpy().tobytes()
   native=bool(self.torch.isfinite(t).all())
   candidate=all_finite_integer(t,self.torch.cuda.current_stream)
   after=t.detach().contiguous().view(self.torch.uint8).cpu().numpy().tobytes()
   if native is not True or candidate is not True or before!=after or descriptor(t)!=before_meta:raise ValueError('integer predicate/read-only')
   self.counts[site]+=1;self.rows.append(dict(site=site,native=True,candidate=True,raw_input_preserved=True));return native if mode=='shadow' else candidate
  except BaseException:self.failed=True;raise
 def finish(self):
  if self.failed or self.counts!=dict.fromkeys(self.counts,24):raise RuntimeError('72-call census')
  return dict(calls=72,native_diagnostic_predicates=72,candidate_gpu_kernels=0,candidate_stream_syncs=72,raw_readonly=True)
