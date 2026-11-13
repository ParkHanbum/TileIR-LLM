import argparse,json,os,fcntl,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
def order():return [(f'{n}_{str(v).lower()}_offset{o}',a,r)for n,v in ((1696,False),(32,False),(0,True))for o in (0,13)for a in ('native','tile')for r in (0,1)]
def result_gate(r):
 rows=r.get('records',[])
 keys=('raw_equal','outside_unchanged','return_identity','layout','repeat_exact','passed')
 ok=r.get('complete')is True and [(x['case'],x['arm'],x['repeat'])for x in rows]==order() and all(all(x.get(k)is True for k in keys)for x in rows)
 return dict(raw_pass=ok,raw_difference_hold=r.get('raw_difference_hold',False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha;m=json.loads(a.manifest.read_text())
 for k in ('source','observations'):assert sha(m[k])==m[k+'_sha256']
 assert [c['name']for c in m['cases']]==[x[0]for x in order()[::4]]
 for c in m['cases']:
  assert c['length'] in (0,32,1696) and type(c['value'])is bool and (not c['value'] or c['length']==0) and c['offset'] in (0,13) and c['owner_bytes']==c['length']+c['offset']+19
 if not a.run_gpu:print(json.dumps(dict(executed=False,calls=24,scope=m['scope'])));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.stat('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock');f=os.fstat(fd);assert(st.st_dev,st.st_ino)==(f.st_dev,f.st_ino);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for env,n in [('CUDA_TILE_CACHE_DIR','tile'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/n).mkdir();os.environ[env]=str(a.output/n)
 r=dict(complete=False,gpu_executed=False,records=[],manifest_sha=a.manifest_sha,scope=m['scope'],timing=False)
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 signal.alarm(285);save()
 try:
  import torch
  from tileir_vllm_runner_bool_fill_v1.api import fill_padding_
  torch.set_num_threads(4);flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k];assert flags()==[True,True]
  for c in m['cases']:
   n,o,size=c['length'],c['offset'],c['owner_bytes'];previous={}
   for arm in ('native','tile'):
    for repeat in (0,1):
     # Alternating canonical bool bytes ensure prefix/suffix tests catch either fill value.
     before=(torch.arange(size,dtype=torch.int64)%2==0);expected=before.clone();expected[o:o+n]=c['value']
     owner=before.to('cuda');view=owner[o:o+n];ptr=owner.untyped_storage().data_ptr();r['gpu_executed']=True;save()
     with torch.inference_mode():out=view.fill_(c['value']) if arm=='native' else fill_padding_(view,c['value'])
     torch.cuda.synchronize();actual=owner.cpu();raw=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8));outside=torch.equal(actual[:o].view(torch.uint8),before[:o].view(torch.uint8)) and torch.equal(actual[o+n:].view(torch.uint8),before[o+n:].view(torch.uint8))
     identity=out is view;layout=out.dtype==torch.bool and out.shape==(n,) and out.stride()==(1,) and out.storage_offset()==o and out.untyped_storage().nbytes()==size and out.untyped_storage().data_ptr()==ptr and owner.untyped_storage().data_ptr()==ptr
     rep=repeat==0 or torch.equal(actual.view(torch.uint8),previous[arm].view(torch.uint8));passed=raw and outside and identity and layout and rep
     r['records'].append(dict(case=c['name'],arm=arm,repeat=repeat,raw_equal=raw,outside_unchanged=outside,return_identity=identity,layout=layout,repeat_exact=rep,empty_no_launch_contract='Source n==0 early return; CPU import-barrier tested, not separate profiler evidence' if n==0 else None,passed=passed));save()
     if not passed:torch.save(dict(before=before,actual=actual,expected=expected,case=c),a.output/'FIRST-FAILURE.pt');r['raw_difference_hold']=True;save();return 2
     previous[arm]=actual
  assert flags()==[True,True];verify_pins(R/'CPU-PINS.json');r['complete']=True;save();return 0
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
