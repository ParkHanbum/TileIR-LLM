import argparse,json,os,fcntl,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
def result_gate(r):
 rows=r.get('records',[]);order=[(c,a,n)for c in ('cu_num_logits','expanded_local_pos')for a in ('native','tile')for n in (0,1)]
 ok=r.get('complete') is True and [(x['case'],x['arm'],x['repeat'])for x in rows]==order and all(x['passed']for x in rows)
 return dict(raw_pass=ok,raw_difference_hold=r.get('raw_difference_hold',False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha;m=json.loads(a.manifest.read_text())
 for k in ('source','trace','observations'):assert sha(m[k])==m[k+'_sha256']
 assert [c['name']for c in m['cases']]==['cu_num_logits','expanded_local_pos']
 if not a.run_gpu:print(json.dumps(dict(executed=False,calls=8,scope=m['scope'])));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.stat('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock');f=os.fstat(fd);assert(st.st_dev,st.st_ino)==(f.st_dev,f.st_ino);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for env,n in [('CUDA_TILE_CACHE_DIR','tile'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/n).mkdir();os.environ[env]=str(a.output/n)
 r=dict(complete=False,gpu_executed=False,records=[],manifest_sha=a.manifest_sha,scope=m['scope'],timing=False)
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 signal.alarm(285);save()
 try:
  import torch
  from tileir_vllm_runner_constructors_v1 import cu_num_logits,expanded_local_pos
  torch.set_num_threads(4);flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k];assert flags()==[True,True]
  for c in m['cases']:
   name=c['name'];expected=torch.arange(33,dtype=torch.int32)if name=='cu_num_logits'else torch.zeros(32,dtype=torch.int32);previous={};live=[]
   for arm in ('native','tile'):
    for repeat in (0,1):
     r['gpu_executed']=True;save()
     with torch.inference_mode():
      if arm=='native':out=torch.arange(33,dtype=torch.int32,device='cuda')if name=='cu_num_logits'else torch.zeros(32,dtype=torch.int32,device='cuda')
      else:out=(cu_num_logits if name=='cu_num_logits'else expanded_local_pos)(32,'cuda')
     torch.cuda.synchronize();actual=out.cpu();raw=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8));layout=out.dtype==torch.int32 and list(out.shape)==c['shape']and out.stride()==(1,)and out.storage_offset()==0 and out.untyped_storage().nbytes()==expected.numel()*4
     fresh=out.untyped_storage().data_ptr()not in[t.untyped_storage().data_ptr()for t in live];prior_unchanged=all(torch.equal(t.cpu().view(torch.uint8),expected.view(torch.uint8))for t in live);rep=repeat==0 or torch.equal(actual,previous[arm]);passed=raw and layout and fresh and prior_unchanged and rep
     r['records'].append(dict(case=name,arm=arm,repeat=repeat,raw_equal=raw,integer_exact=raw,strict_failures=int((actual!=expected).sum()),layout=layout,fresh_disjoint=fresh,earlier_outputs_unchanged=prior_unchanged,repeat_exact=rep,passed=passed));save()
     if not passed:torch.save(dict(actual=actual,expected=expected),a.output/'FIRST-FAILURE.pt');r['raw_difference_hold']=True;save();return 2
     previous[arm]=actual;live.append(out)
  assert flags()==[True,True];verify_pins(R/'CPU-PINS.json');r['complete']=True;save();return 0
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
