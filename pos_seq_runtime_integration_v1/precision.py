"""Record-only precision policy; no flag mutations."""
ORDERS=[]
class PrecisionPolicy:
 def __init__(self,torch):
  self.matmul=torch.backends.cuda.matmul;self.original=self.matmul.allow_bf16_reduced_precision_reduction;self.original_split_k=self.matmul.allow_bf16_reduced_precision_reduction_split_k;self.record=dict(initial_allow_bf16_reduced_precision_reduction=self.original,initial_allow_bf16_reduced_precision_reduction_split_k=self.original_split_k,changes=[],restored=False)
 def select(self,arm):
  assert self.matmul.allow_bf16_reduced_precision_reduction==self.original and self.matmul.allow_bf16_reduced_precision_reduction_split_k==self.original_split_k
  return 'native_unchanged',self.original
 def restore(self):
  self.select(None);self.record.update(restored=True,restored_split_k=True)
