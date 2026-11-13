from . import api
def run(torch):
 def raw(x):return x.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 a=torch.tensor(([-2147483648,-1,0,1,2147483647]*7)[:32],dtype=torch.int32,device='cuda')
 b=torch.tensor(([1,2147483647,-1,-2147483648,0]*7)[:32],dtype=torch.int32,device='cuda')
 before=(raw(a),raw(b));count=0
 for x,y in ((a,b),(b,a)):
  assert raw(api.subtract(x,y))==raw(x-y);count+=1
 assert raw(api.positive(a))==raw(a>0);count+=1
 for _ in range(2):
  actual=api.full((2048,),-1,dtype=torch.int32,device=a.device)
  assert raw(actual)==raw(torch.full((2048,),-1,dtype=torch.int32,device=a.device));count+=1
 assert (raw(a),raw(b))==before and count==5
 return dict(cases=count,passed=True)
