"""Parent-only bounded 71-case raw qualification; default is a CPU-only plan.

Preserves final cat source pins. No model integration or timing. A child gets
580 seconds of work and at most20 seconds of termination budget: its total
lease is600 seconds. Incomplete compilation fails rather than dropping cases.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
PYTHON=Path('/home/n/Documents/Codex/2026-10-31/task/tools/cutilegpt-env/bin/python')
HELPER=Path('/home/n/work/TileIR/cublas-gemm-analysis-20261111/runtime-cubin-probe/launch_probe.py')
COMPILER=Path('/home/n/Documents/Codex/2026-10-31/task/tools/cuda-tileiras-13.4.92/nvidia/cu13/bin/tileiras')
FROZEN=PROJECT/'tileir_cat_v1'
CANDIDATE=PROJECT/'tileir_cat_prepared_v1'
CORE_PINS_SHA256='6be9fe49d9f2a97c76a203a5e6f379f5e81d7e4a07378956e8b46cdf8389a9c9'
HELPER_SHA256='bc956326f1c9f80c1ead01ab071ab18a48ae6a2b4e8c5ee26171789a4b931c0d'
COMPILER_SHA256='8733d2ef6ae234ba5b415d185c957b8cc37577507a2850313a8c0e5c7c3ac54d'
WORK_SECONDS=580
TERMINATE_SECONDS=10
KILL_SECONDS=10
COMPILER_SECONDS=90


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

from tileir_cat_prepared_qual_v1.raw_gate import expected_case_keys,audit_result,CANDIDATE_PINS_SHA256,CANDIDATE_API


def check_pins():
    if sha(FROZEN/'SOURCE-PINS-v2.json')!=CORE_PINS_SHA256:raise RuntimeError('final core manifest changed')
    if sha(CANDIDATE/'SOURCE-PINS.json')!=CANDIDATE_PINS_SHA256:raise RuntimeError('candidate manifest changed')
    core=json.loads((ROOT/'QUALIFICATION-PINS.json').read_text())
    launch=json.loads((ROOT/'LAUNCH-PINS.json').read_text())
    if launch['core_manifest_sha256']!=CORE_PINS_SHA256:raise RuntimeError('wrong launch core pin')
    if launch['candidate_manifest_sha256']!=CANDIDATE_PINS_SHA256:raise RuntimeError('wrong candidate pin')
    files={**core['files'],**launch['files']}
    for name,digest in files.items():
        if sha(name)!=digest:raise RuntimeError('source pin mismatch: '+name)
    if sha(HELPER)!=HELPER_SHA256 or sha(COMPILER)!=COMPILER_SHA256:raise RuntimeError('admission/compiler pin mismatch')
    return files,core,launch


def build_plan(name):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,40}',name):raise ValueError('use a short plain fresh run name')
    run=ROOT/name
    command=[str(PYTHON),'-B',str(ROOT/'qualify_gpu_v3.py'),'--run-gpu','--scope','all','--output',str(run/'RESULT.json')]
    return dict(executed=False,run_directory=str(run),command=command,expected_cases=71,
        work_timeout_seconds=WORK_SECONDS,termination_budget_seconds=TERMINATE_SECONDS+KILL_SECONDS,
        maximum_child_lease_seconds=600,compiler_timeout_seconds=COMPILER_SECONDS,
        note='Parent GPU admission required. No skipped cases, timing or model execution.')


def child_environment(run,fd):
    env={k:v for k,v in os.environ.items() if not k.startswith(('TILEIR_','CUDA_TILE_')) and k not in ('LD_PRELOAD','LD_DEBUG','ALL_GEMM_LOCK_FD')}
    for name in ('tmp','compile','cache','tileir','bytecode','cuda-cache','xdg-cache','torch-cache'):
        (run/name).mkdir(exist_ok=False)
    env.update(PYTHONPATH=str(PROJECT),PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(run/'tmp'),
        LD_LIBRARY_PATH='/usr/local/cuda-13.3/lib64',CUDA_HOME='/usr/local/cuda-13.3',CUDA_PATH='/usr/local/cuda-13.3',
        PATH=str(COMPILER.parent)+':/usr/local/cuda-13.3/bin:'+env.get('PATH',''),
        CUDA_TILE_COMPILER_TIMEOUT_SEC=str(COMPILER_SECONDS),CUDA_TILE_TEMP_DIR=str(run/'compile'),
        CUDA_TILE_CACHE_DIR=str(run/'cache'),CUDA_TILE_DUMP_TILEIR=str(run/'tileir'),
        CUDA_TILE_DUMP_BYTECODE=str(run/'bytecode'),CUDA_CACHE_PATH=str(run/'cuda-cache'),
        XDG_CACHE_HOME=str(run/'xdg-cache'),TORCHINDUCTOR_CACHE_DIR=str(run/'torch-cache'),
        ALL_GEMM_LOCK_FD=str(fd),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    return env


def group_members(pgid):
    members=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==pgid:members.append(dict(pid=int(entry.name),state=fields[0],start_ticks=fields[19]))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return members


def stop_child(child,record):
    if child is None:return
    # Only this launcher's new process group may be signaled. Never kill an
    # unrecognized GPU process found by the shared admission helper.
    if child.poll() is None or group_members(child.pid):
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        deadline=time.monotonic()+TERMINATE_SECONDS
        while time.monotonic()<deadline:
            child.poll()
            if child.returncode is not None and not group_members(child.pid):break
            time.sleep(.1)
        if child.poll() is None or group_members(child.pid):
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            deadline=time.monotonic()+KILL_SECONDS
            while time.monotonic()<deadline:
                child.poll()
                if child.returncode is not None and not group_members(child.pid):break
                time.sleep(.1)
    record['returncode']=child.poll()
    record['remaining_process_group']=group_members(child.pid)
    record['child_absent']=not Path(f'/proc/{child.pid}').exists()


def execute(plan):
    files,core,launch=check_pins()
    run=Path(plan['run_directory']);run.mkdir(exist_ok=False)
    record=dict(plan,executed=True,started=time.time(),core_manifest_sha256=CORE_PINS_SHA256,
        candidate_manifest_sha256=CANDIDATE_PINS_SHA256,candidate_api=CANDIDATE_API,
        launch_pins_sha256=sha(ROOT/'LAUNCH-PINS.json'),launcher_sha256=sha(__file__))
    child=None
    def save():(run/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    spec=importlib.util.spec_from_file_location('cat_admission_only',HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    try:
        with open(helper.LOCK,'r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            record['lock_path']=helper.LOCK
            stat=os.fstat(lock.fileno());record['lock_device_inode']=[stat.st_dev,stat.st_ino]
            try:
                record['pre']=helper.admit();save()
                env=child_environment(run,lock.fileno())
                with (run/'stdout').open('x') as stdout,(run/'stderr').open('x') as stderr:
                    child=subprocess.Popen(plan['command'],cwd=PROJECT,env=env,stdout=stdout,stderr=stderr,
                        pass_fds=(lock.fileno(),),start_new_session=True)
                    record['pid']=child.pid;record['child_started']=time.time();save()
                    try:record['returncode']=child.wait(timeout=WORK_SECONDS)
                    except subprocess.TimeoutExpired:record['timeout']=True
            finally:
                stop_child(child,record)
                if child is not None:record['child_finished']=time.time()
                record.setdefault('child_absent',True)
                record.setdefault('remaining_process_group',[])
                if record.get('pre') is not None and record['child_absent'] and not record['remaining_process_group']:
                    try:record['after']=helper.admit()
                    except BaseException as error:record['recovery_error']=repr(error)
                fcntl.flock(lock,fcntl.LOCK_UN)
        with open(helper.LOCK,'r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            record['lock_reacquired']=True
            fcntl.flock(lock,fcntl.LOCK_UN)
    except BaseException as error:
        record['error']=repr(error)
        if child is not None and child.poll() is None:stop_child(child,record)
    finally:
        mismatches={}
        for name,digest in files.items():
            try:actual=sha(name)
            except OSError as error:actual=repr(error)
            if actual!=digest:mismatches[name]=actual
        record['post_pin_mismatches']=mismatches
        result_path=run/'RESULT.json'
        if result_path.exists():
            record['result_sha256']=sha(result_path)
            try:
                result=json.loads(result_path.read_text());record['raw_gate']=audit_result(result)
                if result.get('pins')!=core:record['raw_gate']['errors'].append('child_source_manifest_mismatch');record['raw_gate']['passed']=False
            except (ValueError,TypeError,KeyError) as error:record['result_error']=repr(error)
        else:record['raw_gate']=dict(passed=False,errors=['no_result'])
        record['success']=bool(record.get('returncode')==0 and record.get('raw_gate',{}).get('passed') and
            record.get('child_absent') and not record.get('remaining_process_group') and record.get('lock_reacquired') and
            'after' in record and not any(record.get(k) for k in ('timeout','error','recovery_error','post_pin_mismatches','result_error')))
        record['finished']=time.time();save()
    print(json.dumps(record,indent=2));return 0 if record['success'] else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--run-name',default='catprepared01')
    args=parser.parse_args(argv)
    try:plan=build_plan(args.run_name)
    except ValueError as error:parser.error(str(error))
    if not args.run_gpu:
        print(json.dumps(plan,indent=2));return 0
    return execute(plan)

if __name__=='__main__':raise SystemExit(main())
