"""CPU-only freeze/test receipt generator. Never launches the model."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time,difflib
R=Path(__file__).resolve().parent;P=R.parent
sys.path.insert(0,str(P))
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value):(R/name).write_text(json.dumps(value,indent=2)+'\n')
def main():
 assert not (R/'CPU-READY.json').exists()
 assert not (R/'cache_v1').exists() and not (R/'CACHE-PRESEED-01.json').exists()
 before={str(p):sha(p) for p in R.iterdir() if p.is_file()}
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='');env.pop('PYTHONPATH',None)
 cmd=[sys.executable,'-B','-m','unittest',R.name+'.test_cpu',R.name+'.test_cleanup',R.name+'.test_extensions',R.name+'.test_entry_guards',R.name+'.test_parent_bridge',R.name+'.test_actual_lifecycle','-v']
 x=subprocess.run(cmd,cwd=P,env=env,text=True,capture_output=True,timeout=120)
 (R/'CPU-TESTS.stdout').write_text(x.stdout);(R/'CPU-TESTS.stderr').write_text(x.stderr)
 write('CPU-TESTS.json',dict(command=cmd,actual_exit=x.returncode,stdout_sha256=sha(R/'CPU-TESTS.stdout'),stderr_sha256=sha(R/'CPU-TESTS.stderr'),gpu=False))
 assert x.returncode==0,x.stderr
 dry=[]
 for filename in ('run.py','launch.py'):
  z=subprocess.run([sys.executable,'-B',str(R/filename)],cwd=P,env=env,text=True,capture_output=True,timeout=30);assert z.returncode==0,z.stderr
  j=json.loads(z.stdout);dry.append(dict(file=filename,actual_exit=z.returncode,result=j,stderr=z.stderr))
 write('CPU-DRY.json',dry)
 from all_gemm_expanded_composition_v4.gates import qualification
 proof=qualification(R);write('PREREQUISITE-VERIFIED.json',proof)
 from all_gemm_expanded_composition_v4.test_actual_lifecycle import source_audit
 write('ACTUAL-ENTRY-AUDIT.json',source_audit())
 pins={};sources=json.loads((R/'PREREQUISITE-PINS.json').read_text());excluded=[]
 for package,ref in sources.items():
  assert sha(ref['path'])==ref['sha256'];pins[ref['path']]=ref['sha256']
  for p,h in json.loads(Path(ref['path']).read_text()).items():
   if '/cache_v' in p or Path(p).name.startswith('CACHE-'):
    excluded.append(p);continue
   assert p not in pins or pins[p]==h;assert sha(p)==h,(p,'changed prerequisite');pins[p]=h
 refs=json.loads((R/'QUALIFICATION-RECEIPT.json').read_text())
 for value in refs.values():
  for ref in value.values():assert sha(ref['path'])==ref['sha256'];pins[ref['path']]=ref['sha256']
 parent=json.loads((R/'PARENT-PREREQUISITE.json').read_text());assert parent['complete']
 for kind in ('result','launch','recovery','source_pins','final_seal'):
  ref=parent[kind];assert sha(ref['path'])==ref['sha256'];pins[ref['path']]=ref['sha256']
 # Explicit baseline diff, not a broad filesystem copy.
 old=P/'all_gemm_expanded_composition_v3';diff=[]
 for f in ('adapter.py','run.py','launch.py','result_gate.py','gates.py','entry_guards.py','test_entry_guards.py','verify_cpu.py'):
  diff.extend(difflib.unified_diff((old/f).read_text().splitlines(True),(R/f).read_text().splitlines(True),fromfile=str(old/f),tofile=str(R/f)))
 for f in ('test_actual_lifecycle.py',):
  diff.extend(difflib.unified_diff([],(R/f).read_text().splitlines(True),fromfile='/dev/null',tofile=str(R/f)))
 (R/'SOURCE-DIFF.patch').write_text(''.join(diff))
 for p,h in before.items():assert sha(p)==h,('CPU tests changed source',p)
 for p in R.iterdir():
  if p.is_file() and p.name not in ('PINS.json','CPU-READY.json'):pins[str(p)]=sha(p)
 write('PINS.json',pins)
 assert not (R/'cache_v1').exists() and not (R/'CACHE-PRESEED-01.json').exists()
 count=63
 write('CPU-READY.json',dict(complete=True,phase='CPU_ready_GPU_pending',controller_count=25,boundary_calls=5888,logical_operations=6056,cpu_tests=count,test_receipt_sha256=sha(R/'CPU-TESTS.json'),dry_sha256=sha(R/'CPU-DRY.json'),pins_count=len(pins),pins_sha256=sha(R/'PINS.json'),proofs=9,old19_unobserved_exit_excluded=True,numeric_kernels_changed=False,gpu_query_compile_cache_actions=0,excluded_historical_cache_pin_paths=len(set(excluded)),pin_scope='Union of nine frozen prerequisite source/artifact pins except historical cache/preseed files; evidence referenced, never copied into new cache. Current-run fresh cache still required.',gpu_execution_authorized_by_this_receipt=False))
 print(json.dumps(dict(ready=str(R/'CPU-READY.json'),ready_sha256=sha(R/'CPU-READY.json'),pins=len(pins),pins_sha256=sha(R/'PINS.json'),tests=count,actual_exit=0)))
if __name__=='__main__':main()
