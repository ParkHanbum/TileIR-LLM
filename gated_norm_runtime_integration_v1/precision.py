"""Read-only original policy observation; no setters."""
ORDERS=[]
class PrecisionPolicy:
 def __init__(self,torch):
  self.m=torch.backends.cuda.matmul;self.a=self.m.allow_bf16_reduced_precision_reduction;self.b=self.m.allow_bf16_reduced_precision_reduction_split_k
  self.record=dict(initial_allow_bf16_reduced_precision_reduction=self.a,initial_allow_bf16_reduced_precision_reduction_split_k=self.b,changes=[],restored=False)
 def select(self,arm):
  assert self.m.allow_bf16_reduced_precision_reduction is self.a and self.m.allow_bf16_reduced_precision_reduction_split_k is self.b
  return 'unchanged_native',self.a
 def restore(self):
  self.select('observe');self.record.update(restored=True,restored_split_k=True)
