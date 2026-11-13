"""Read-only native default precision policy. Never assigns backend flags."""
ORDERS=[]
class PrecisionPolicy:
 def __init__(self,torch):
  self.matmul=torch.backends.cuda.matmul
  self.original=self.matmul.allow_bf16_reduced_precision_reduction
  self.original_split_k=self.matmul.allow_bf16_reduced_precision_reduction_split_k
  assert self.original is True and self.original_split_k is True,'Native initial flags are not True/True; do not alter them'
  self.record=dict(policy='native_default_true_true_read_only',initial_allow_bf16_reduced_precision_reduction=self.original,initial_allow_bf16_reduced_precision_reduction_split_k=self.original_split_k,changes=[],observations=[],restored=False,restored_split_k=False)
 def select(self,arm):
  assert arm=='DEFAULT'
  actual=self.matmul.allow_bf16_reduced_precision_reduction;split=self.matmul.allow_bf16_reduced_precision_reduction_split_k
  assert actual is True and split is True,'Unexpected flag drift; not overwritten'
  self.record['observations'].append(dict(arm=arm,reduction=actual,split_k=split))
  return 'native_default_unchanged',actual
 def restore(self):
  assert self.matmul.allow_bf16_reduced_precision_reduction is self.original and self.matmul.allow_bf16_reduced_precision_reduction_split_k is self.original_split_k,'Flag drift on cleanup; not overwritten'
  self.record.update(restored=True,restored_split_k=True)
