"""Observe and preserve caller's native policy; never change GEMM settings."""
ORDERS=()
class PrecisionPolicy:
 def __init__(self,torch):
  self.torch=torch;self.initial=self.values();assert self.initial==dict(bf16=True,split_k=True);self.record=dict(policy='unchanged native',initial=self.initial,restored=False,restored_split_k=False)
 def values(self):
  m=self.torch.backends.cuda.matmul
  return dict(bf16=m.allow_bf16_reduced_precision_reduction,split_k=m.allow_bf16_reduced_precision_reduction_split_k)
 def select(self,arm):
  assert arm=='native' and self.values()==self.initial
  return 'native',self.initial['bf16']
 def restore(self):
  assert self.values()==self.initial
  self.record.update(restored=True,restored_split_k=True)
