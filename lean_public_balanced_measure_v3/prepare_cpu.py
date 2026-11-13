"""CPU-only final source/test freeze after actual lean25 prerequisite completion."""
from pathlib import Path
import ast,difflib,hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;P=R.parent;Q=P/'lean_public_runtime_qualification_v2';OLD=P/'lean_public_balanced_measure_v2'
sys.path.insert(0,str(P))
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
 p=R/name;assert not p.exists();p.write_text(json.dumps(obj,indent=2)+'\n');return sha(p)
def main():
 assert not(R/'CPU-READY.json').exists()and not(R/'PINS.json').exists()
 assert not(R/'process0').exists()and not(R/'process1').exists()
 from lean_public_balanced_measure_v3.gates import qualification
 proof=qualification(R)
 files=('__init__.py','design.py','scopes.py','request.py','profile_contract.py','gates.py','run.py','launch.py','test_cpu.py','test_contract_cpu.py','test_pending_cpu.py','prepare_cpu.py')
 for name in files:ast.parse((R/name).read_text())
 diff=[]
 for name in files:
  old=OLD/name
  diff.extend(difflib.unified_diff(old.read_text().splitlines(True)if old.exists()else[],(R/name).read_text().splitlines(True),fromfile=str(old)if old.exists()else'/dev/null',tofile=str(R/name)))
 (R/'SOURCE-DIFF.patch').write_text(''.join(diff))
 before={str(p):sha(p)for p in R.iterdir()if p.is_file()}
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(P))
 cmd=[sys.executable,'-B','-m','unittest','lean_public_balanced_measure_v3.test_cpu','lean_public_balanced_measure_v3.test_contract_cpu','lean_public_balanced_measure_v3.test_pending_cpu','-v']
 start=time.time();p=subprocess.run(cmd,cwd=P,env=env,capture_output=True,timeout=120)
 (R/'CPU-FULL.stdout').write_bytes(p.stdout);(R/'CPU-FULL.stderr').write_bytes(p.stderr)
 after={n:sha(n)for n in before}
 receipt=dict(complete=p.returncode==0 and before==after,actual_exit=p.returncode,seconds=time.time()-start,command=cmd,source_unchanged=before==after,stdout_sha256=sha(R/'CPU-FULL.stdout'),stderr_sha256=sha(R/'CPU-FULL.stderr'),gpu=False)
 th=save('CPU-TESTS.json',receipt);assert receipt['complete'],p.stderr.decode()
 dh=[]
 for index in(0,1):
  for name in('run.py','launch.py'):
   z=subprocess.run([sys.executable,'-B',str(R/name),'--process-index',str(index)],cwd=P,env=env,capture_output=True,text=True,timeout=30);assert z.returncode==0,z.stderr
   dh.append(dict(file=name,process_index=index,actual_exit=z.returncode,result=json.loads(z.stdout)))
 drysha=save('CPU-DRY.json',dh);proofsha=save('ACTUAL-PREREQUISITE-VERIFIED.json',proof)
 pins=json.loads((Q/'PINS.json').read_text());excluded=[]
 for p in list(pins):
  if '/cache_v' in p or Path(p).name.startswith('CACHE-'):excluded.append(p);pins.pop(p)
 for p,h in pins.items():assert sha(p)==h,('Changed prerequisite',p)
 for name in files:
  p=OLD/name
  if p.exists():pins[str(p)]=sha(p)
 # Original uninstrumented timer source used by the retained AST regression.
 extra=[P/'lean_public_balanced_measure_v1/request.py',Q/'PINS.json',Q/'CPU-READY.json']
 extra += [Path(v['path'])for k,v in proof['evidence'].items()if isinstance(v,dict)and 'path'in v]
 extra += [p for p in R.iterdir()if p.is_file()]
 for p in extra:pins[str(p)]=sha(p)
 ph=save('PINS.json',pins)
 rh=save('CPU-READY.json',dict(complete=True,phase='CPU_balanced_measurement_preparation_complete_GPU_pending',pins_count=len(pins),pins_sha256=ph,cpu_tests_sha256=th,dry_sha256=drysha,actual_prerequisite_verified_sha256=proofsha,actual_cpu_test_exit=0,review_required=True,parent_cache_preseed_required=True,gpu_executed=False,gpu_query=False,cuda_compile=False,cache_created=False,processes=2,pairs_per_process=4,profiles_per_process=2,controllers=25,boundaries=5888,logical_operations=6056,fixed_scope='Completed lean25 endpoint proof plus full25 contract proof; old19 excluded; public API guards retained',baseline='current eager mode0/graphsNONE B32P53G16logprobs5defaultT/T',excluded_historical_cache_pins=len(excluded)))
 print(json.dumps(dict(complete=True,pins=len(pins),pins_sha256=ph,ready_sha256=rh,cpu_tests_sha256=th,actual_exit=0)))
if __name__=='__main__':main()
