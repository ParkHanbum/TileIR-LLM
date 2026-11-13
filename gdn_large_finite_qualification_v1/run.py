"""CPU-default large-v3 synthetic qualification; first failure stop, no retries."""
import argparse,json,os,fcntl,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from gdn_large_finite_qualification_v1.cases import cases,expected,guarded_target
def result_gate(r):
 try:
  assert r['complete']is True and r['gpu_executed']is True and not r.get('error') and not r.get('cleanup_error')
  assert r['precision_before']==r['precision_after']==[True,True]
  assert [(x['dtype'],x['n'],x['variant'])for x in r['records']]==cases()
  for x in r['records']:
   assert x['expected']is expected(x['variant']) and x['passed']is True
   assert x['native']==x['tile']==[x['expected']]*2 and x['host_target_calls']==2*int(x['expected'])
   assert x['input_bytes_preserved'] and x['metadata_preserved'] and x['same_current_stream'] and x['private_launches']==4
  return dict(raw_pass=True,raw_difference_hold=False)
 except (AssertionError,KeyError,TypeError):return dict(raw_pass=False,raw_difference_hold=bool(r.get('raw_difference_hold')))
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha;m=json.loads(a.manifest.read_text());assert m['cases']==[list(x)for x in cases()]
 if not a.run_gpu:print(json.dumps(dict(executed=False,records=32,scope=m['scope'])));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.stat('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock');f=os.fstat(fd);assert(st.st_dev,st.st_ino)==(f.st_dev,f.st_ino);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for env,n in [('CUDA_TILE_CACHE_DIR','tile'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/n).mkdir();os.environ[env]=str(a.output/n)
 r=dict(complete=False,gpu_executed=False,records=[],manifest_sha=a.manifest_sha,scope=m['scope'],timing=False,automatic_retry=False)
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 def timeout(*_):raise TimeoutError('585s child limit')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(585);save();torch=None
 try:
  import torch,cuda.tile as ct
  from tileir_validation_primitives_v3.api import all_finite
  from tileir_validation_primitives_v3.kernels import count_nonfinite,reduce_counts
  from finite_guard_qualification_v1.spy import launch_spy
  torch.set_num_threads(4)
  flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
  r['precision_before']=flags();assert flags()==[True,True]
  r.update(gpu_executed=True,phase='synthetic_edges');save()
  def meta(x):return [tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.data_ptr(),x.untyped_storage().nbytes(),x.is_neg(),x.is_conj()]
  for dtype,n,variant in cases():
   r['current_case']=dict(dtype=dtype,n=n,variant=variant);save()
   # One GPU payload at a time, <=64MiB; snapshots are host bytes.
   x=torch.zeros(n,dtype=getattr(torch,dtype.split('.')[-1]),device='cuda')
   values={'finite':1.0,'negative_zero':-0.0,'subnormal':torch.finfo(x.dtype).tiny/2,'maxfinite':torch.finfo(x.dtype).max,'nan':float('nan'),'negative_nan':-float('nan'),'inf':float('inf'),'negative_inf':-float('inf')}
   x[-1]=values[variant];before=x.view(torch.uint8).cpu();metadata=meta(x);native=[];tile=[];target=[];launches=0
   stream=torch.cuda.current_stream(x.device).cuda_stream
   for repeat in range(2):
    value=bool(torch.isfinite(x).all());native.append(value);assert value is expected(variant)
    assert torch.equal(before,x.view(torch.uint8).cpu()) and meta(x)==metadata
   for repeat in range(2):
    events=[]
    with launch_spy(ct,(),(count_nonfinite,reduce_counts),events)as counts:
     value=guarded_target(x,all_finite,lambda:target.append('after_host_status'))
    tile.append(value);assert counts==dict(target=0,guard=2,other=0)
    blocks=(n+1023)//1024;width=1<<(blocks-1).bit_length()
    assert events[0]['grid']==(blocks,) and events[1]['grid']==(1,) and events[1]['args'][-1]==width==16384
    launches+=counts['guard'];del events
    assert value is expected(variant) and torch.equal(before,x.view(torch.uint8).cpu()) and meta(x)==metadata
    assert torch.cuda.current_stream(x.device).cuda_stream==stream
   row=dict(dtype=dtype,n=n,variant=variant,expected=expected(variant),native=native,tile=tile,host_target_calls=len(target),private_launches=launches,input_bytes_preserved=True,metadata_preserved=True,same_current_stream=True,passed=True)
   r['records'].append(row);save();del x,before
  torch.cuda.synchronize();r['precision_after']=flags();assert r['precision_after']==r['precision_before'];r.update(complete=True,phase='complete');save();assert result_gate(r)['raw_pass'];return 0
 except BaseException as e:
  r.update(error=repr(e),traceback=traceback.format_exc(),complete=False,status='HOLD',automatic_retry=False);save();return 1
 finally:
  signal.alarm(0)
  if torch is not None:
   try:
    if r['gpu_executed']:torch.cuda.synchronize()
   except BaseException as e:
    r['cleanup_error']=repr(e);r['complete']=False;save();raise RuntimeError('Final CUDA synchronization failed')from e
if __name__=='__main__':raise SystemExit(main())
