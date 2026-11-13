def run():
 import torch
 from tileir_vllm_sampler_integer_v1.api import gather_integer_rows,new_sampled_ones
 records=[]
 for dtype in (torch.int32,torch.int64):
  for n in (32,1696):
   x=torch.arange(n,dtype=dtype,device='cuda')-n;idx=torch.tensor([-n,-1,0,n-1]*8,dtype=torch.int64,device='cuda');old=x.cpu();oi=idx.cpu();out=[]
   for repeat in range(2):
    native=x[idx];candidate=gather_integer_rows(x,idx)
    assert torch.equal(native.view(torch.uint8),candidate.view(torch.uint8))and torch.equal(old,x.cpu())and torch.equal(oi,idx.cpu())
    assert all(candidate.data_ptr()!=t.data_ptr()for t in [x,idx,native,*out]);out.append(candidate)
   records.append(dict(n=n,dtype=str(dtype),repeats=2,raw_exact=True,input_preserved=True,fresh=True))
 for fill in (0,-1):
  x=torch.full((32,),fill,dtype=torch.int32,device='cuda');out=[]
  for repeat in range(2):
   native=x.new_ones(32);candidate=new_sampled_ones(x,32)
   assert torch.equal(native,candidate)and bool((x==fill).all())and all(candidate.data_ptr()!=t.data_ptr()for t in [x,native,*out]);out.append(candidate)
  records.append(dict(n=32,dtype='torch.int32',repeats=2,raw_exact=True,input_preserved=True,fresh=True))
 return dict(passed=True,records=records,scope='Synthetic negative/duplicate index and ones, separate from actual48')
