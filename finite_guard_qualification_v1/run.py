import argparse,json,os,fcntl,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
def result_gate(r):
 rows=r.get('records',[])
 ok=r.get('complete')is True and len(rows)==41 and all(x.get('passed')is True for x in rows) and sum(x['kind']=='actual'for x in rows)==3 and sum(x['kind']=='invalid_top5'for x in rows)==12 and sum(x['kind']=='generic'for x in rows)==24 and sum(x['kind']=='negative_view'for x in rows)==2
 ok=ok and len({x.get('bank')for x in rows if x['kind']=='actual'})==3
 ok=ok and {(x.get('value'),x.get('index'))for x in rows if x['kind']=='invalid_top5'}=={(v,i)for v in ('nan','negative_nan','inf','negative_inf')for i in (0,1024,7946239)}
 ok=ok and {(x.get('dtype'),x.get('n'),x.get('variant'))for x in rows if x['kind']=='generic'}=={(d,n,v)for d in ('torch.bfloat16','torch.float32')for n in (1,1023,1024,1025)for v in ('finite','nan','inf')}
 ok=ok and {x.get('invalid')for x in rows if x['kind']=='negative_view'}=={False,True}
 return dict(raw_pass=ok,raw_difference_hold=r.get('raw_difference_hold',False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha;m=json.loads(a.manifest.read_text());assert sha(m['source'])==m['source_sha256'];assert len(m['actual_banks'])==3
 for b in m['actual_banks']:assert sha(b['path'])==b['sha256']
 if not a.run_gpu:print(json.dumps(dict(executed=False,scope=m['scope'],records_expected=41)));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.stat('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock');f=os.fstat(fd);assert(st.st_dev,st.st_ino)==(f.st_dev,f.st_ino);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for env,n in [('CUDA_TILE_CACHE_DIR','tile'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/n).mkdir();os.environ[env]=str(a.output/n)
 r=dict(complete=False,gpu_executed=False,records=[],manifest_sha=a.manifest_sha,scope=m['scope'],timing=False)
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 signal.alarm(585);save()
 try:
  import torch,cuda.tile as ct
  from tileir_validation_primitives_v2.api import require_finite
  from tileir_validation_primitives_v2.kernels import count_nonfinite,reduce_counts
  from tileir_vllm_top5_v2.api import top5_indices as original
  from tileir_vllm_top5_v2.kernels import blocks,finish
  from finite_guard_qualification_v1.candidate import top5_indices as candidate
  from finite_guard_qualification_v1.spy import launch_spy
  torch.set_num_threads(4);flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k];assert flags()==[True,True]
  def snapshot(x):
   raw=torch.empty(0,dtype=torch.uint8,device=x.device).set_(x.untyped_storage(),0,(x.untyped_storage().nbytes(),),(1,)).cpu().clone()
   return raw,(x.is_neg(),x.is_conj(),x.data_ptr(),x.storage_offset(),tuple(x.shape),tuple(x.stride()))
  def unchanged(x,s):
   after=snapshot(x);return torch.equal(after[0],s[0])and after[1]==s[1]
  def guard(x):
   events=[];current=torch.cuda.current_stream(x.device).cuda_stream
   with launch_spy(ct,(blocks,finish),(count_nonfinite,reduce_counts),events)as counts:
    try:require_finite(x);ok=True;error=None
    except AssertionError as e:ok=False;error=str(e)
   assert counts==dict(target=0,guard=2,other=0),counts
   assert torch.cuda.current_stream(x.device).cuda_stream==current
   b=(x.numel()+1023)//1024;w=1<<(b-1).bit_length()
   assert [e['grid']for e in events]==[(b,),(1,)] and events[1]['args'][-1]==w
   assert w>=b and w-b<w and b*1024-x.numel()<1024
   return ok,error
  def record(row,x=None,s=None):
   if x is not None:row['input_owner_flags_unchanged']=unchanged(x,s);row['passed']=row['passed']and row['input_owner_flags_unchanged']
   r['records'].append(row);save()
   if not row['passed']:
    if x is not None:torch.save(dict(input_snapshot=s[0],actual_snapshot=snapshot(x)[0],row=row),a.output/'FIRST-FAILURE.pt')
    r['raw_difference_hold']=True;save();raise ArithmeticError('First correctness HOLD')
  r['gpu_executed']=True;save()
  for bank in m['actual_banks']:
   x=torch.load(bank['path'],map_location='cpu',weights_only=True).to('cuda');s=snapshot(x);assert x.dtype==torch.bfloat16 and x.shape==(32,248320)
   stream=torch.cuda.current_stream(x.device).cuda_stream;native_finite=bool(torch.isfinite(x).all());g=[guard(x)[0]for _ in range(2)];outs=[];launches=[]
   for fn in (original,original,candidate,candidate):
    with launch_spy(ct,(blocks,finish),(count_nonfinite,reduce_counts))as counts:outs.append(fn(x))
    launches.append(dict(counts))
   torch.cuda.synchronize();equal=all(torch.equal(o.cpu().view(torch.uint8),outs[0].cpu().view(torch.uint8))for o in outs[1:])
   passed=native_finite and g==[True,True] and equal and launches[:2]==[dict(target=2,guard=0,other=0)]*2 and launches[2:]==[dict(target=2,guard=2,other=0)]*2 and torch.cuda.current_stream(x.device).cuda_stream==stream
   record(dict(kind='actual',bank=bank['path'],native_finite=native_finite,guard_repeat=g,top5_raw_equal=equal,launches=launches,passed=passed),x,s)
  # Actual candidate top5 path: guard failure must happen before blocks/finish.
  for valname,value in [('nan',float('nan')),('negative_nan',-float('nan')),('inf',float('inf')),('negative_inf',-float('inf'))]:
   for index in (0,1024,7946239):
    x=torch.zeros((32,248320),device='cuda',dtype=torch.bfloat16);x.view(-1)[index]=value;s=snapshot(x);observed=[]
    for fn in (original,candidate):
     with launch_spy(ct,(blocks,finish),(count_nonfinite,reduce_counts))as counts:
      try:fn(x);error=None
      except AssertionError as e:error=(type(e).__name__,str(e))
     observed.append(dict(error=error,counts=dict(counts)))
    passed=all(z['error']==('AssertionError','NaN/Inf outside finite-only contract')for z in observed)and observed[0]['counts']==dict(target=0,guard=0,other=0)and observed[1]['counts']==dict(target=0,guard=2,other=0)
    record(dict(kind='invalid_top5',value=valname,index=index,observed=observed,passed=passed),x,s)
  for dtype in (torch.bfloat16,torch.float32):
   for n in (1,1023,1024,1025):
    for variant in ('finite','nan','inf'):
     x=torch.zeros(n,device='cuda',dtype=dtype)
     if variant=='finite':
      x[0]=-0.0
      if n>1:x[1]=torch.finfo(dtype).tiny/2
      if n>2:x[-1]=torch.finfo(dtype).max
     else:x[-1]=float(variant)
     s=snapshot(x);expected=bool(torch.isfinite(x).all());g=[guard(x)[0]for _ in range(2)];record(dict(kind='generic',dtype=str(dtype),n=n,variant=variant,expected=expected,guard_repeat=g,passed=g==[expected,expected]),x,s)
  for invalid in (False,True):
   real=torch.zeros(513,dtype=torch.float32,device='cuda');imag=torch.ones(513,dtype=torch.float32,device='cuda')
   if invalid:real[-1]=float('nan')
   z=torch.complex(real,imag);x=z.conj().imag.as_strided((1025,),(1,),0);assert x.is_neg() and x.dtype==torch.float32
   s=snapshot(x);stream=torch.cuda.current_stream(x.device).cuda_stream;expected=bool(torch.isfinite(x).all());g=[guard(x)[0]for _ in range(2)]
   record(dict(kind='negative_view',invalid=invalid,expected=expected,guard_repeat=g,passed=g==[expected,expected] and torch.cuda.current_stream(x.device).cuda_stream==stream),x,s)
  assert flags()==[True,True];verify_pins(R/'CPU-PINS.json');r['complete']=True;save();return 0
 except ArithmeticError:return 2
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
