import hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;Q=P/'lean_public_runtime_qualification_v1';V1=P/'lean_public_balanced_measure_v1'
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
 p=R/name;assert not p.exists();p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');return sha(p)
names=['__init__.py','design.py','scopes.py','request.py','profile_contract.py','gates.py','run.py','launch.py','test_cpu.py','test_contract_cpu.py','prepare_cpu.py','PLAN.md','SCOPE-CORRECTION.md','SOURCE-DIFF.patch','QUALIFICATION-RECEIPT.json'];before={str(R/n):sha(R/n)for n in names};pins=json.loads((V1/'PINS.json').read_text());assert all(sha(Path(p))==h for p,h in pins.items())
assert not(R/'process0').exists()and not(R/'process1').exists()
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(P)};cmd=[sys.executable,'-B','-m','unittest','lean_public_balanced_measure_v2.test_cpu','lean_public_balanced_measure_v2.test_contract_cpu','-v'];t=time.time();p=subprocess.run(cmd,cwd=P,env=env,capture_output=True,timeout=60)
(R/'cpu0.stdout').write_bytes(p.stdout);(R/'cpu0.stderr').write_bytes(p.stderr);after={str(R/n):sha(R/n)for n in names}
test=dict(complete=p.returncode==0 and before==after,actual_exit=p.returncode,timed_out=False,seconds=time.time()-t,argv=cmd,source_before=before,source_after=after,source_unchanged=before==after,stdout_sha256=sha(R/'cpu0.stdout'),stderr_sha256=sha(R/'cpu0.stderr'),scope='CPU-only real helper mocks, exact output/evidence gates, balanced schedules, profiler-scope restore and actual dry imports; no GPU/CUDA compile/model/cache')
th=save('CPU-TESTS.json',test);assert test['complete'],p.stderr.decode()
assert all(sha(Path(q))==h for q,h in pins.items())
extra=[V1/'PINS.json',V1/'CPU-READY.json',Q/'PINS.json',Q/'CPU-READY.json',Q/'leanCorrectness01/RESULT.json',Q/'leanCorrectness01-launch.json',Q/'leanCorrectness01-recovery.json']+[R/n for n in names+['CPU-TESTS.json','cpu0.stdout','cpu0.stderr']]
for q in extra:pins[str(q)]=sha(q)
ph=save('PINS.json',pins);rh=save('CPU-READY.json',dict(complete=True,phase='CPU_balanced_measurement_preparation_complete_GPU_pending',pins_count=len(pins),pins_sha256=ph,cpu_tests_sha256=th,actual_cpu_test_exit=0,review_required=True,parent_cache_preseed_required=True,gpu_executed=False,gpu_query=False,cuda_compile=False,cache_created=False,processes=2,pairs_per_process=4,profiles_per_process=2,fixed_scope='Prior18controllers only; uncertain19notadopted',baseline='current eager mode0/graphsNONE B32P53G16logprobs5defaultT/T'))
print(json.dumps(dict(complete=True,pins=len(pins),pins_sha256=ph,ready_sha256=rh,cpu_tests_sha256=th,actual_exit=p.returncode)))
