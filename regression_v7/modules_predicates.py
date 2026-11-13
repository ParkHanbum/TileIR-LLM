"""Representative original-v7 public predicates, no optimized variants."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parents[1]
def cpu_tests():return ['public_eq_all_guard_v2.test_cpu','public_eq_all_guard_v2.test_bits','tileir_validation_primitives_v2.test_cpu','regression_v7.test_predicates']
def dependencies():
 paths=[P/'tileir_validation_primitives_v3'/n for n in ('api.py','kernels.py','contract.py')]+[P/'public_eq_all_guard_v2'/n for n in ('api.py','kernel.py','contract.py')]
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
def expected_rows():
 return [dict(kind='finite',dtype=d,case=c,expected=c not in ('nan','inf','negative_inf')) for d in ('torch.bfloat16','torch.float32') for c in ('finite','negative_zero','subnormal','nan','inf','negative_inf')]+[dict(kind='equality',constant=k,case=c,expected=c in ('contiguous','broadcast_offset')) for k in (0,1) for c in ('contiguous','broadcast_offset','nan','wrong')]
def audit(r):
 try:
  expected=expected_rows();rows=r['records']
  assert r['complete'] is True and len(rows)==len(expected)
  for row,spec in zip(rows,expected):
   assert all(row[k]==v and type(row[k])is type(v) for k,v in spec.items())
   assert row['passed']is True and row['readonly']is True
   assert row['native']is row['candidate']is spec['expected']
  return True
 except (KeyError,TypeError,AssertionError):return False
def run_gpu(torch,output):
 from tileir_validation_primitives_v3.api import all_finite
 from public_eq_all_guard_v2.api import eq_all
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 result=dict(complete=False,records=[],scope='private predicate edges; not full workload')
 def save():(output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 def raw(x):return x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
 cases=[('finite',0.,True),('negative_zero',-0.,True),('subnormal',None,True),('nan',float('nan'),False),('inf',float('inf'),False),('negative_inf',float('-inf'),False)]
 try:
  for dtype in (torch.bfloat16,torch.float32):
   for name,value,expected in cases:
    x=torch.zeros(1025,dtype=dtype,device='cuda');x[-1]=torch.finfo(dtype).tiny/2 if value is None else value
    before=raw(x);stream=torch.cuda.current_stream(x.device).cuda_stream
    native=bool(torch.isfinite(x).all());got=all_finite(x)
    row=dict(kind='finite',dtype=str(dtype),case=name,native=native,candidate=got,expected=expected,readonly=raw(x)==before,passed=False);result['records'].append(row);save()
    assert native is got is expected and row['readonly'] and torch.cuda.current_stream(x.device).cuda_stream==stream
    row['passed']=True;save()
  for constant in (0,1):
   for kind in ('contiguous','broadcast_offset','nan','wrong'):
    owner=torch.full((1025,),float(constant),device='cuda');x=owner
    if kind=='broadcast_offset':x=owner[3:4].reshape(()).expand(32,4)
    if kind=='nan':owner[-1]=float('nan')
    if kind=='wrong':owner[-1]=float(1-constant)
    expected=kind in ('contiguous','broadcast_offset');before=raw(owner);stream=torch.cuda.current_stream(x.device).cuda_stream
    native=bool((x==constant).all());got=eq_all(x,constant)
    row=dict(kind='equality',case=kind,constant=constant,native=native,candidate=got,expected=expected,readonly=raw(owner)==before,passed=False);result['records'].append(row);save()
    assert native is got is expected and row['readonly'] and torch.cuda.current_stream(x.device).cuda_stream==stream
    row['passed']=True;save()
  torch.cuda.synchronize();result['complete']=True;assert audit(result);save();return result
 except BaseException:
  result['complete']=False;save();raise
