"""Run CPU-only tests, verify inherited pins, freeze new harness source; no GPU."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;L=P/'lean_public_composition_v1'
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,d):
 p=R/name;assert not p.exists(),str(p);p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');return sha(p)
names=['__init__.py','run.py','gates.py','cleanup.py','launch.py','test_cpu.py','prepare_cpu.py','PLAN.md','QUALIFICATION-RECEIPT.json']
before={str(R/n):sha(R/n)for n in names};pins=json.loads((L/'PINS.json').read_text())
assert all(sha(Path(p))==h for p,h in pins.items())
assert not (R/'cache_v1').exists()and not (R/'CACHE-PRESEED-01.json').exists()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(P));cmd=[sys.executable,'-B','-m','unittest','lean_public_composition_v1.test_core_cpu','lean_public_composition_v1.test_sampler_cpu','lean_public_composition_v1.test_recurrent_cpu','lean_public_runtime_qualification_v1.test_cpu','-v'];t=time.time()
p=subprocess.run(cmd,cwd=P,env=env,capture_output=True,timeout=60);after={str(R/n):sha(R/n)for n in names}
(R/'cpu1.stdout').write_bytes(p.stdout);(R/'cpu1.stderr').write_bytes(p.stderr)
tests=dict(complete=p.returncode==0 and before==after,actual_exit=p.returncode,timed_out=False,seconds=time.time()-t,argv=cmd,tests_expected=64,source_before=before,source_after=after,source_unchanged=before==after,stdout_sha256=sha(R/'cpu1.stdout'),stderr_sha256=sha(R/'cpu1.stderr'),scope='CPU-only mock, source, actual default-import and output/cleanup gates; no GPU/CUDA compile/model/cache')
th=save('CPU-TESTS.json',tests);assert tests['complete'],p.stderr.decode()
assert all(sha(Path(q))==h for q,h in pins.items()),'Frozen lean/prior pins changed during tests'
for q in [L/'PINS.json',L/'CPU-READY.json']+[R/n for n in names+['CPU-TESTS.json','cpu1.stdout','cpu1.stderr']]:pins[str(q)]=sha(q)
ph=save('PINS.json',pins)
ready=dict(complete=True,phase='CPU_harness_preparation_complete_GPU_pending',pins_count=len(pins),pins_sha256=ph,cpu_tests_sha256=th,actual_cpu_test_exit=0,tests=64,parent_cache_preseed_required=True,review_required=True,source_only=True,gpu_execution=False,gpu_query=False,cuda_compile=False,source_pins_preserved=True,baseline='current eager mode0/graphsNONE B32P53G16 logprobs5 defaultT/T',scope='native-to-lean512/public2560 exact +5648boundary/5680logical census/restore; not repeated intermediate raw proof or performance')
rh=save('CPU-READY.json',ready);print(json.dumps(dict(complete=True,pins=len(pins),pins_sha256=ph,ready_sha256=rh,cpu_tests_sha256=th,actual_exit=p.returncode,tests=64)))
