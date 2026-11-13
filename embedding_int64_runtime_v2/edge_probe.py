def run():
 import torch
 from tileir_vllm_embedding_int64_v1.api import embedding_ids_long
 records=[]
 for n in (32,1696):
  values=([-2147483648,-1,0,1,2147483647]*((n+4)//5))[:n]
  x=torch.tensor(values,dtype=torch.int32,device='cuda');before=x.cpu();outputs=[]
  for repeat in range(2):
   native=x.long();out=embedding_ids_long(x)
   assert torch.equal(native.view(torch.uint8),out.view(torch.uint8))and out.cpu().tolist()==values and torch.equal(before,x.cpu())
   assert all(out.data_ptr()!=p.data_ptr()for p in [x,native,*outputs]);outputs.append(out)
  records.append(dict(n=n,repeats=2,raw_exact=True,signed_exact=True,input_preserved=True,fresh=True))
 return dict(passed=True,records=records,scope='Synthetic signed edges, not actual token bank')
