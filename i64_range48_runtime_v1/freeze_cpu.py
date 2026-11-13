import json,pathlib,hashlib,subprocess,sys,os
R=pathlib.Path(__file__).resolve().parent;P=R.parent
PY='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 assert sys.executable==PY,'Use runtime venv'
 assert not(R/'PINS.json').exists()and not(R/'CPU-READY.json').exists(),'Already frozen'
 assert not any(p.name in ('cache_v1','__pycache__')or p.suffix in ('.pyc','.cubin','.bytecode','.pt')for p in R.rglob('*'))
 pins={}
 for src in ('eq408_runtime_v1/PINS.json','i64_range_synthetic_qualification_v1/CPU-PINS.json'):
  for path,h in json.loads((P/src).read_text()).items():
   assert path not in pins or pins[path]==h;assert sha(path)==h,('Inherited drift',path);pins[path]=h
 for f in R.iterdir():
  if f.is_file():pins[str(f)]=sha(f)
 # New API model callbacks refer only to immutable already-qualified numerical API.
 (R/'PINS.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
 env=dict(os.environ,PYTHONPATH=str(P),PYTHONDONTWRITEBYTECODE='1');checks=[]
 commands=[[PY,'-B','-m','unittest','i64_range48_runtime_v1.test_cpu'],[PY,'-B',str(R/'run.py')],[PY,'-B',str(R/'launch.py')],[PY,'-B','-c',"from pathlib import Path;from i64_range48_runtime_v1.gates import qualification;import sys;assert qualification(Path('"+str(R)+"'))['passed'];assert 'torch' not in sys.modules and 'cuda.tile' not in sys.modules;print('PREREQUISITES PASS; framework imports0')"]]
 for cmd in commands:
  p=subprocess.run(cmd,cwd=P,env=env,text=True,capture_output=True,timeout=120);checks.append(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,checks[-1]
 (R/'CPU-TESTS.json').write_text(json.dumps(checks,indent=2)+'\n')
 for path,h in pins.items():assert sha(path)==h
 assert not any(p.name in ('cache_v1','__pycache__')or p.suffix in ('.pyc','.cubin','.bytecode','.pt')for p in R.rglob('*'))
 ready=dict(status='CPU_READY_FOR_INDEPENDENT_REVIEW',tests=5,dry_entries=2,actual_prerequisite_gate=True,pins=len(pins),pins_sha256=sha(R/'PINS.json'),tests_sha256=sha(R/'CPU-TESTS.json'),prerequisite_sha256=sha(R/'PREREQUISITE.json'),source_hashes_reverified=True,gpu_query_compile_cache=False,actual_range48_pending=True,sampling_overlay_inactive=True)
 (R/'CPU-READY.json').write_text(json.dumps(ready,indent=2)+'\n');print(json.dumps(ready))
if __name__=='__main__':main()
