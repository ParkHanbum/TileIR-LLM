"""Parent-owned eager position/sequence-length correctness capture; one process per invocation."""
import fcntl, hashlib, importlib.util, json, os, signal, subprocess, sys, time
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
sys.path.insert(0,str(P))
from tileir_cat_prepared_qual_v1.launch_raw_all_v1 import stop_child,group_members
PYTHON='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
H=Path('/home/n/work/TileIR/cublas-gemm-analysis-20261111/runtime-cubin-probe/launch_probe.py')
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 assert len(sys.argv)==2 and sys.argv[1].isalnum()
 name=sys.argv[1];run=R/name;receipt=R/(name+'-launch.json');assert not run.exists() and not receipt.exists()
 assert sha(H)=='bc956326f1c9f80c1ead01ab071ab18a48ae6a2b4e8c5ee26171789a4b931c0d'
 spec=importlib.util.spec_from_file_location('admission_only',H);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 pins=json.loads((R/'PINS.json').read_text());rec={'pins':pins,'launcher_sha256':sha(__file__),'scope':'Correctness-only eager public position/sequence-length capture; compile/graph off; one unobserved and one observed token-only request'}
 def save():receipt.write_text(json.dumps(rec,indent=2)+'\n')
 child=None
 with open(h.LOCK,'r') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   for p,d in pins.items():assert sha(p)==d,(p,'changed source')
   rec['pre']=h.admit();save()
   env={k:v for k,v in os.environ.items() if not k.startswith(('TILEIR_','CUDA_TILE_')) and k not in ('LD_PRELOAD','LD_DEBUG')}
   cache=str(R/'cache_v1')
   driver_cache=R/'cache_v1'/'cuda';driver_cache.mkdir(parents=True,exist_ok=True)
   temp=R/(name+'-tmp');temp.mkdir(exist_ok=False)
   env.update(CUDA_CACHE_PATH=str(driver_cache),TMPDIR=str(temp),TMP=str(temp),TEMP=str(temp))
   cuda='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/nvidia/cu13'
   env.pop('LD_LIBRARY_PATH',None)
   env.update(CUDA_HOME=cuda,CUDA_PATH=cuda,PATH='/home/n/Documents/Codex/2026-10-31/task/tools/cuda-tileiras-13.4.92/nvidia/cu13/bin:'+cuda+'/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')
   for key in list(env):
    if key.startswith(('VLLM_','TORCHINDUCTOR_','TRITON_')):env.pop(key)
   env.pop('PYTHONPATH',None)
   env.update(PYTHONDONTWRITEBYTECODE='1',ALL_GEMM_LOCK_FD=str(lock.fileno()),VLLM_COMPARISON_PARENT_APPROVED='1',XDG_CACHE_HOME=cache,VLLM_CACHE_ROOT=cache+'/vllm',TRITON_CACHE_DIR=cache+'/triton',TORCHINDUCTOR_CACHE_DIR=cache+'/inductor',FLASHINFER_WORKSPACE_BASE=cache+'/flashinfer')
   cmd=[PYTHON,'-B',str(R/'run.py'),'--run-gpu','--output',str(run)];rec['command']=cmd;rec['correctness_only']=True
   with (R/(name+'.stdout')).open('x') as out,(R/(name+'.stderr')).open('x') as err:
    child=subprocess.Popen(cmd,env=env,cwd=P,stdout=out,stderr=err,pass_fds=(lock.fileno(),),start_new_session=True);rec['pid']=child.pid;save()
    try:rec['returncode']=child.wait(timeout=930)
    except subprocess.TimeoutExpired:
     rec['timeout']=True;os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     rec['returncode']=child.returncode
   stop_child(child,rec)
   for p,d in pins.items():assert sha(p)==d,(p,'changed during run')
   if (run/'RESULT.json').exists():
    rec['result_sha256']=sha(run/'RESULT.json');result=json.loads((run/'RESULT.json').read_text())
    samples=result.get('samples',[]);warm=result.get('warmup_records',[])
    all_requests=warm+samples
    from result_gate import audit
    rec['result_gate']=audit(result);rec['measurement_complete']=rec['result_gate']['passed']
   assert rec.get('child_absent') and not rec.get('remaining_process_group'),'Owned process group not empty'
   rec['after']=h.admit()
  except BaseException as e:
   rec['error']=repr(e)
   if child is not None and child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    rec['returncode']=child.returncode
  finally:
   if child is not None:stop_child(child,rec)
   rec['finished']=time.time();save()
 with open(h.LOCK,'r') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rec['lock_reacquired']=True
 save();print(json.dumps(rec,indent=2))
 return 0 if rec.get('returncode')==0 and rec.get('measurement_complete') and rec.get('child_absent') and rec.get('lock_reacquired') and not rec.get('remaining_process_group') and 'after' in rec and not rec.get('timeout') and 'error' not in rec else 1
if __name__=='__main__':raise SystemExit(main())
