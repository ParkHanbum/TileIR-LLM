from .qualifier import Qualifier

def run(torch):
 q=Qualifier(torch,cap=48)
 for dtype,values in [(torch.uint64,[0,2**63,2**64-1,1]),(torch.int64,[-2**63,2**63-1,0,-1])]:
  for n in (1,2,3,4):
   a=torch.tensor(values[:n],dtype=dtype,device='cuda')
   if q.pair(a,a.clone()) is not True:raise ValueError('synthetic equal')
   for lane in range(n):
    for bit in (0,63):
     changed=values[:n];bits=(changed[lane] & ((1<<64)-1))^(1<<bit)
     changed[lane]=bits-(1<<64) if dtype==torch.int64 and bits>=(1<<63) else bits
     b=torch.tensor(changed,dtype=dtype,device='cuda')
     if q.pair(a,b) is not False:raise ValueError('synthetic differing high/low bit')
 if len(q.rows)!=48 or q.failed:raise RuntimeError('synthetic census')
 return dict(cases=48,equal_cases=8,unequal_cases=40,all_inputs_readonly=True,passed=True)
