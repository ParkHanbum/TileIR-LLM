"""Public Tensor.record_stream lifetime, same numeric enqueue as qualified Copy."""
from gdn_prefill_output_copy_v3.api import Copy as OriginalCopy
class Copy(OriginalCopy):
 def __init__(self):super().__init__();self.recorded=False
 def __call__(self,dst,src):
  if self.failed:raise RuntimeError('Copy failure latched')
  try:
   import torch
   before=torch.cuda.current_stream(dst.device).cuda_stream
   result=super().__call__(dst,src)
   if len(self.pending)!=1:raise RuntimeError('Unexpected pending copy owners')
   d,s,sv,dv,stream=self.pending[0]
   if before!=stream.cuda_stream or torch.cuda.current_stream(dst.device).cuda_stream!=before:raise RuntimeError('Copy stream changed')
   # Strong references stay alive through launch and every recording operation.
   # Record views too, although their allocation aliases the source/destination.
   for tensor in(d,s,sv,dv):tensor.record_stream(stream)
   if torch.cuda.current_stream(dst.device).cuda_stream!=before:raise RuntimeError('Record stream changed')
   self.recorded=True;self.pending.clear()
   return result
  except BaseException:self.failed=True;raise
