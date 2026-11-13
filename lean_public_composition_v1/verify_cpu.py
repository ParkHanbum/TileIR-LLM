"""Bounded CPU tests and source receipt; no model/CUDA compile/GPU inspection."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
FILES=['__init__.py','support.py','model.py','metadata.py','recurrent.py','sampler.py','adapter.py','test_core_cpu.py','test_sampler_cpu.py','test_recurrent_cpu.py','verify_cpu.py','PLAN.md','DATAFLOW-AUDIT.md','QUALIFICATION-PLAN.md']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pins():return {str(R/n):sha(R/n) for n in FILES}
before=pins();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(P));argv=[sys.executable,'-B','-m','unittest','lean_public_composition_v1.test_core_cpu','lean_public_composition_v1.test_sampler_cpu','lean_public_composition_v1.test_recurrent_cpu','-v'];begin=time.time()
assert not (R/'CPU-TESTS.json').exists()
try:
 result=subprocess.run(argv,cwd=P,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60);rc=result.returncode;out=result.stdout;err=result.stderr;timeout=False
except subprocess.TimeoutExpired as e:rc=None;out=e.stdout or b'';err=e.stderr or b'';timeout=True
(R/'cpu1.stdout').write_bytes(out);(R/'cpu1.stderr').write_bytes(err);after=pins()
receipt=dict(complete=rc==0 and before==after,actual_exit=rc,timed_out=timeout,seconds=time.time()-begin,argv=argv,cwd=str(P),cuda_visible_devices='',tests_expected=29,source_before=before,source_after=after,source_unchanged=before==after,stdout_sha256=sha(R/'cpu1.stdout'),stderr_sha256=sha(R/'cpu1.stderr'),scope='CPU mocks/source checks only; no CUDA initialization, GPU query, CUDA compile, model or numerical proof',warning='Previous cpu0 stderr records24 testsOK but command-exit receipt not retained; this final29-test run records actual child exit')
(R/'CPU-TESTS.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k not in('source_before','source_after')}));sys.exit(0 if receipt['complete'] else 1)
