"""CPU default; bounded private GPU qualifier, no model/performance."""
import argparse,json,os,fcntl,signal,sys
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from sampling_guard_synthetic_qualification_v1.cases import cases,expected
from sampling_guard_synthetic_qualification_v1.prerequisite import verify
def result_gate(r):
 try:
  plan=cases();assert r['complete']is True and not r.get('error')and len(r['records'])==len(plan)+2
  for number,(row,spec)in enumerate(zip(r['records'],plan)):
   assert row['kind']=='classification'and row['case']==number and row['spec']==spec and row['passed']is True
   assert row['native']==row['candidate']==[expected(spec)]*2 and row['readonly_checks']==4
  for row,dtype in zip(r['records'][-2:],['torch.float32','torch.bfloat16']):
   assert row['kind']=='gather'and row['dtype']==dtype and row['native']==row['candidate']==[True,True]and row['passed']is True and row['readonly_checks']==4
  assert r['final_synchronize']is True and r['range_prerequisite']['passed']is True
  return dict(raw_pass=True,cases=len(plan)+2,scope='private sampling guard primitives; model binding pending')
 except(AssertionError,KeyError,TypeError):return dict(raw_pass=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha
 prereq=verify()
 if not a.run_gpu:print(json.dumps(dict(executed=False,cases=len(cases())+2,model_calls=0,range_prerequisite=prereq)));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);s=os.stat('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock');f=os.fstat(fd);assert(s.st_dev,s.st_ino)==(f.st_dev,f.st_ino);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for key,name in [('CUDA_TILE_CACHE_DIR','tile'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/name).mkdir();os.environ[key]=str(a.output/name)
 result=dict(complete=False,records=[],manifest_sha=a.manifest_sha,model_calls=0,timing=False,range_prerequisite=prereq)
 def save():(a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 def timeout(*_):raise TimeoutError('585s worker bound')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(585);save()
 try:
  import torch
  assert torch.__version__=='2.13.0+cu130'
  from sampling_guard_synthetic_qualification_v1.probe import run
  run(torch,save,result)
  torch.cuda.synchronize();result['final_synchronize']=True;result['complete']=True;save()
 except BaseException as e:result['error']=repr(e);save();raise
 finally:signal.alarm(0)
 return 0 if result_gate(result)['raw_pass']else 1
if __name__=='__main__':raise SystemExit(main())
