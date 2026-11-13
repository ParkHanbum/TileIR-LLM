"""Runtime-only specialization/edge check, not performance; imports inside run only."""
def run():
 import torch
 from tileir_validation_primitives_v2.api import require_finite
 rows=[]
 for dtype,n in ((torch.float32,7946240),(torch.bfloat16,81920),(torch.bfloat16,32)):
  x=torch.zeros(n,dtype=dtype,device='cuda');x[0]=-0.0;x[-1]=torch.finfo(dtype).tiny/2
  for label,pos,value in [('finite',None,None),('nan_first',0,float('nan')),('posinf_last',n-1,float('inf')),('neginf_tail',n-1,-float('inf'))]:
   if pos is not None:x[pos]=value
   before=x.view(torch.uint8).cpu().clone();stream=torch.cuda.current_stream();reference=bool(torch.isfinite(x).all());verdict=[]
   for repeat in range(2):
    try:require_finite(x);ok=True
    except AssertionError as e:
     assert str(e)=='NaN/Inf outside finite-only contract';ok=False
    assert ok==reference;assert torch.cuda.current_stream()==stream
    assert torch.equal(before,x.view(torch.uint8).cpu());verdict.append(ok)
   rows.append(dict(dtype=str(dtype),n=n,label=label,repeats=verdict,reference=reference,input_raw_preserved=True))
   if pos is not None:x[pos]=0
 assert len(rows)==12
 return dict(passed=True,rows=rows,scope='Synthetic new specializations/edges; actual model six bindings checked separately')
