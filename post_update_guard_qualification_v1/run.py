import argparse,json,os,fcntl,signal,sys
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from public_post_update_guard_v1.cases import cases
from public_post_update_guard_v1.contract import MESSAGES

def result_gate(result):
 try:
  plan=cases();assert result['complete']is True and not result.get('error')and len(result['records'])==len(plan)
  for row,(label,_,expected)in zip(result['records'],plan):
   assert row['case']==label and row['expected']==expected and row['native']==row['candidate']==[expected]*2
   assert row['passed']is True and row['readonly_checks']==4 and row['stream_unchanged']is True
   count=9 if expected is None else (8 if label=='delta_overflow'else MESSAGES.index(expected)+1)
   assert row['candidate_stages']==[list(range(count))]*2
  assert result['final_synchronize']is True
  return dict(raw_pass=True,cases=len(plan),scope='Private device backing, original-address/model proof pending')
 except(AssertionError,KeyError,TypeError):return dict(raw_pass=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha
 if not a.run_gpu:print(json.dumps(dict(executed=False,cases=len(cases()),model_calls=0)));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.stat('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock');f=os.fstat(fd);assert(st.st_dev,st.st_ino)==(f.st_dev,f.st_ino);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for key,name in [('CUDA_TILE_CACHE_DIR','tile'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/name).mkdir();os.environ[key]=str(a.output/name)
 result=dict(complete=False,records=[],manifest_sha=a.manifest_sha,model_calls=0,timing=False)
 def save():(a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 def progress(rows):result['records']=rows;save()
 def timeout(*_):raise TimeoutError('585s worker bound')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(585);save()
 try:
  import torch
  from public_post_update_guard_v1.qualify import qualify
  assert torch.__version__=='2.13.0+cu130'
  proof=qualify(torch,progress)
  assert proof['complete'];torch.cuda.synchronize();result.update(complete=True,final_synchronize=True);save()
 except BaseException as e:result['error']=repr(e);save();raise
 finally:signal.alarm(0)
 return 0 if result_gate(result)['raw_pass']else 1
if __name__=='__main__':raise SystemExit(main())
