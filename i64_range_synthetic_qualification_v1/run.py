import argparse,json,os,fcntl,signal,sys
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from public_i64_range_guard_v1.cases import cases
from public_i64_range_guard_v1.contract import reference

def result_gate(r):
 try:
  plan=cases();assert r['complete']is True and not r.get('error') and len(r['records'])==2*len(plan)
  for i,row in enumerate(r['records']):
   spec=plan[i%len(plan)];assert row['spec']==spec and row['stream']==i//len(plan) and row['case']==i%len(plan)
   assert row['stage']=='verified' and row['passed']is True and type(row['native'])is type(row['candidate'])is bool
   assert row['native']is row['candidate']is reference(spec['values'],spec['lower'],spec['upper'])
   assert row['readonly_checks']==2 and row['stream_unchanged']is True
  assert r['final_synchronize']is True and r['current_stream_restored']is True and r['model_calls']==0 and r['timing']is False
  return dict(raw_pass=True,cases=2*len(plan),scope='private synthetic only; model48 pending')
 except (AssertionError,KeyError,TypeError):return dict(raw_pass=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha
 assert json.loads(a.manifest.read_text())==json.loads((R/'MANIFEST.json').read_text())
 if not a.run_gpu:print(json.dumps(dict(executed=False,cases=2*len(cases()),model_calls=0)));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.stat('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock');f=os.fstat(fd);assert(st.st_dev,st.st_ino)==(f.st_dev,f.st_ino);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for key,name in [('CUDA_TILE_CACHE_DIR','tile'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/name).mkdir();os.environ[key]=str(a.output/name)
 result=dict(complete=False,records=[],manifest_sha=a.manifest_sha,model_calls=0,timing=False)
 def save():(a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 def record(row):
  idx=row['stream']*len(cases())+row['case']
  if idx==len(result['records']):result['records'].append(dict(row))
  else:assert idx==len(result['records'])-1;result['records'][idx]=dict(row)
  save()
 def timeout(*_):raise TimeoutError('585s worker bound')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(585);save()
 try:
  import torch
  assert torch.__version__=='2.13.0+cu130'
  from public_i64_range_guard_v1.qualifier import qualify
  result.update(qualify(torch,record));save()
 except BaseException as e:result['error']=repr(e);save();raise
 finally:signal.alarm(0)
 return 0 if result_gate(result)['raw_pass']else 1
if __name__=='__main__':raise SystemExit(main())
