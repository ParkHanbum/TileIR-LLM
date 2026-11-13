"""Process-local public Torch precision policy, not a vendor hook."""
ARMS={'D':('baseline',True),'F':('baseline',False),'T':('tileir',False)}
ORDERS=[['D','F','T'],['F','T','D'],['T','D','F'],['T','F','D'],['F','D','T'],['D','T','F']]
class PrecisionPolicy:
 def __init__(self,torch):
  self.matmul=torch.backends.cuda.matmul;self.original=self.matmul.allow_bf16_reduced_precision_reduction
  self.original_split_k=self.matmul.allow_bf16_reduced_precision_reduction_split_k
  assert type(self.original) is bool and type(self.original_split_k) is bool
  self.record={'initial_allow_bf16_reduced_precision_reduction':self.original,'initial_allow_bf16_reduced_precision_reduction_split_k':self.original_split_k,'changes':[],'restored':False}
 def select(self,arm):
  backend,setting=ARMS[arm];self.matmul.allow_bf16_reduced_precision_reduction=(setting,True)
  actual=self.matmul.allow_bf16_reduced_precision_reduction;assert actual is setting
  split_k=self.matmul.allow_bf16_reduced_precision_reduction_split_k;assert split_k is True
  self.record['changes'].append({'arm':arm,'backend':backend,'allow_bf16_reduced_precision_reduction':actual,'allow_bf16_reduced_precision_reduction_split_k':split_k})
  return backend,actual
 def restore(self):
  self.matmul.allow_bf16_reduced_precision_reduction=(self.original,self.original_split_k)
  assert self.matmul.allow_bf16_reduced_precision_reduction is self.original
  assert self.matmul.allow_bf16_reduced_precision_reduction_split_k is self.original_split_k
  self.record['restored_split_k']=True;self.record['restored']=True
