"""CPU-default large-v3 synthetic qualification; first failure stop, no retries."""
import argparse,json,os,fcntl,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from kv_strided_finite_qualification_v1.probe import CASES,run as probe

def result_gate(r):
 try:
  assert r['complete']is True and r['gpu_executed']is True and not r.get('error') and not r.get('cleanup_error')
  assert r['precision_before']==r['precision_after']==[True,True]
  expected=[(p,c,a,i)for p in ('prefill','decode')for c in CASES for a in ('native','tile')for i in range(2)]
  assert [(x['phase'],x['case'],x['arm'],x['repeat'])for x in r['records']]==expected
  for x in r['records']:
   good=x['case']not in ('first_nan','last_nan','first_posinf','last_neginf')
   assert x['expected']is good and x['value']is good and x['owner_preserved'] and x['passed']
  return dict(raw_pass=True,raw_difference_hold=False)
 except (AssertionError,KeyError,TypeError):return dict(raw_pass=False,raw_difference_hold=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha;m=json.loads(a.manifest.read_text());assert m['cases']==list(CASES)
 if not a.run_gpu:print(json.dumps(dict(executed=False,records=72,scope=m['scope'])));return 0
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
  torch.set_num_threads(4)
  flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
  r['precision_before']=flags();assert flags()==[True,True]
  r.update(gpu_executed=True,phase='strided_bank_and_edges');save()
  def progress(part):
   r['records']=part['records'];r['active_case']=part.get('active_case');save()
  probe(progress)
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
