import argparse,json,os,fcntl,signal,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from eq_all_synthetic_qualification_v1.cases import cases,expected

def result_gate(r):
 try:
  plan=cases();assert r['complete'] is True and not r.get('error') and len(r['records'])==len(plan)
  for row,spec in zip(r['records'],plan):
   assert row['spec']==spec and row['passed'] is True and row['native']==row['candidate']==[expected(spec)]*2
   assert row['readonly_checks']==4 and row['stream_unchanged'] is True
  assert r['final_synchronize'] is True
  return dict(raw_pass=True,cases=len(plan),scope='private synthetic only; model408 pending')
 except (AssertionError,KeyError,TypeError):return dict(raw_pass=False)
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
 def timeout(*_):raise TimeoutError('585s worker bound')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(585);save()
 try:
  import torch
  from public_eq_all_guard_v2.api import eq_all
  assert torch.__version__=='2.13.0+cu130'
  def signed(w):return w if w<0x80000000 else w-(1<<32)
  def snapshot(owner):return owner.detach().view(torch.uint8).cpu().numpy().tobytes()
  for spec in cases():
   n=spec['n'];constant=spec['constant'];kind=spec['kind'];base=0 if constant==0 else 0x3f800000
   if kind=='broadcast':
    owner=torch.full((spec['offset']+2,),signed(base),dtype=torch.int32,device='cuda');owner[spec['offset']]=signed(spec['word'])
    x=owner[spec['offset']:spec['offset']+1].view(torch.float32).reshape(()).expand(32,4)
   else:
    owner=torch.full((n,),signed(base),dtype=torch.int32,device='cuda')
    if n:owner[spec['index']]=signed(spec['word'])
    x=owner.view(torch.float32)
    if kind=='scalar':x=x.reshape(())
   before=snapshot(owner);stream=torch.cuda.current_stream(x.device).cuda_stream
   row=dict(spec=spec,native=[],candidate=[],readonly_checks=0,passed=False,stream_unchanged=False);result['records'].append(row);save()
   for label,fn in [('native',lambda:bool((x==constant).all())),('candidate',lambda:eq_all(x,constant))]:
    for _ in range(2):
     value=fn();assert type(value)is bool;row[label].append(value)
     assert snapshot(owner)==before,'entire owner bytes modified';row['readonly_checks']+=1
     assert torch.cuda.current_stream(x.device).cuda_stream==stream
     assert value is expected(spec),'exact native/candidate/IEEE predicate disagreement'
   row['stream_unchanged']=True;row['passed']=True;save()
  torch.cuda.synchronize();result['final_synchronize']=True;result['complete']=True;save()
 except BaseException as e:result['error']=repr(e);save();raise
 finally:signal.alarm(0)
 return 0 if result_gate(result)['raw_pass'] else 1
if __name__=='__main__':raise SystemExit(main())
