"""Narrow no-draft TP1 runner constructors; fresh outputs, no global patch."""
def validate_num_reqs(n):
 assert type(n)is int and n==32,'Only observed B32 no-draft constructor contract supported'
def _make(n,device,arange):
 validate_num_reqs(n)
 import torch,cuda.tile as ct
 from .kernel import construct
 dev=torch.device(device);assert dev.type=='cuda'
 out=torch.empty((n+1 if arange else n,),dtype=torch.int32,device=dev)
 ct.launch(torch.cuda.current_stream(out.device),(1,),construct,(out,arange))
 return out
def cu_num_logits(num_reqs,device):return _make(num_reqs,device,True)
def expanded_local_pos(num_reqs,device):return _make(num_reqs,device,False)
