"""Parent-owned admission/lock/exit receipt; CPU default, no automatic retries."""
import argparse,fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
sys.path[:0]=[str(P),str(R)]
from packed_grouped_qualification_v1.contract import sha,verify_pins
PYTHON='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
H=Path('/home/n/work/TileIR/cublas-gemm-analysis-20261111/runtime-cubin-probe/launch_probe.py')
H_SHA='bc956326f1c9f80c1ead01ab071ab18a48ae6a2b4e8c5ee26171789a4b931c0d'

from kv_strided_finite_qualification_v1.run import result_gate

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--run-name',required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
    assert a.run_name.isalnum() and len(a.run_name)<=40
    manifest=Path(a.manifest).resolve();assert sha(manifest)==a.manifest_sha
    pins=verify_pins(R/'CPU-PINS.json');assert sha(H)==H_SHA
    run=R/a.run_name;receipt=R/(a.run_name+'-launch.json')
    cmd=[PYTHON,'-B',str(R/'run.py'),'--run-gpu','--output',str(run),'--manifest',str(manifest),'--manifest-sha',a.manifest_sha]
    if not a.run_gpu:print(json.dumps(dict(executed=False,command=cmd,source_pins_checked=True)));return 0
    assert not run.exists() and not receipt.exists()
    from tileir_cat_prepared_qual_v1.launch_raw_all_v1 import stop_child
    spec=importlib.util.spec_from_file_location('admission_only',H);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    rec=dict(manifest=str(manifest),manifest_sha=a.manifest_sha,pins=pins,command=cmd,scope='Strided KV finite bank and synthetic predicate correctness; no model/timing');child=None
    def save():receipt.write_text(json.dumps(rec,indent=2)+'\n')
    with open(h.LOCK,'r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            rec['pre']=h.admit();save()
            env={k:v for k,v in os.environ.items() if not k.startswith(('TILEIR_','CUDA_TILE_')) and k not in ('LD_PRELOAD','LD_DEBUG')}
            cuda='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/nvidia/cu13'
            env.pop('LD_LIBRARY_PATH',None)
            env.update(PYTHONPATH=str(P),PYTHONDONTWRITEBYTECODE='1',CUDA_HOME=cuda,CUDA_PATH=cuda,ALL_GEMM_LOCK_FD=str(lock.fileno()),CUDA_TILE_COMPILER_TIMEOUT_SEC='90',PATH='/home/n/Documents/Codex/2026-10-31/task/tools/cuda-tileiras-13.4.92/nvidia/cu13/bin:'+cuda+'/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')
            with (R/(a.run_name+'.stdout')).open('x') as out,(R/(a.run_name+'.stderr')).open('x') as err:
                child=subprocess.Popen(cmd,cwd=P,env=env,stdout=out,stderr=err,pass_fds=(lock.fileno(),),start_new_session=True);rec['pid']=child.pid;save()
                try:rec['returncode']=child.wait(timeout=600)
                except subprocess.TimeoutExpired:rec['timeout']=True
                finally:stop_child(child,rec)
            verify_pins(R/'CPU-PINS.json');assert sha(manifest)==a.manifest_sha
            if (run/'RESULT.json').exists():
                rec['result_sha256']=sha(run/'RESULT.json');r=json.loads((run/'RESULT.json').read_text());assert r['manifest_sha']==a.manifest_sha;rec['result_gate']=result_gate(r)
            if rec.get('child_absent') and not rec.get('remaining_process_group'):rec['after']=h.admit()
            else:rec['recovery_error']='Owned child/process group remains'
        except BaseException as e:
            rec['error']=repr(e)
            if child is not None:stop_child(child,rec)
        finally:
            if child is not None:
                stop_child(child,rec)
                if rec.get('child_absent') and not rec.get('remaining_process_group') and 'after' not in rec:
                    try:rec['after']=h.admit()
                    except BaseException as e:rec['recovery_error']=repr(e)
            rec['finished']=time.time();save()
    with open(h.LOCK,'r') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rec['lock_reacquired']=True
    save();print(json.dumps(rec,indent=2))
    recovered=rec.get('child_absent') and rec.get('lock_reacquired') and not rec.get('remaining_process_group') and 'after' in rec and not any(rec.get(k) for k in ('error','timeout','recovery_error'))
    gate=rec.get('result_gate',{})
    if recovered and rec.get('returncode')==0 and gate.get('raw_pass'):return 0
    if recovered and rec.get('returncode')==2 and gate.get('raw_difference_hold'):return 2
    return 1
if __name__=='__main__':raise SystemExit(main())
