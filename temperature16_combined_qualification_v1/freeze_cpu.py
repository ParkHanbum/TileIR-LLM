import pathlib,hashlib,json,subprocess,os,sys,ast
R=pathlib.Path(__file__).resolve().parent;P=R.parent
PY='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
def sha(p):
 with pathlib.Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 assert sys.executable==PY
 for n in ('PINS.json','CPU-READY.json','cache_v1','CACHE-PRESEED-01.json'):assert not(R/n).exists()
 base=P/'lean_sampling144_v1/PINS.json';assert sha(base)=='36377a9668d48c0da448aa0e4941399ed96c5fbc6293cd71c9199915a9365f6d'
 pins=json.loads(base.read_text())
 for p,h in pins.items():assert sha(p)==h,p
 pins[str(base)]=sha(base)
 for folder in (R,P/'public_temperature_zero_v1',P/'public_eq_all_guard_v2'):
  for f in folder.iterdir():
   if f.is_file()and f.suffix in('.py','.txt','.json','.md'):
    if f.suffix=='.py':ast.parse(f.read_text())
    pins[str(f)]=sha(f)
 (R/'PINS.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
 commands=[[PY,'-B','-m','unittest','temperature16_combined_qualification_v1.test_cpu','lean_sampling144_v1.test_cpu'],[PY,'-B',str(R/'run.py')],[PY,'-B',str(R/'launch.py')],[PY,'-B','-c','from temperature16_combined_qualification_v1.gates import qualification;from pathlib import Path;print(qualification(Path('+repr(str(R))+')))']]
 results=[]
 for cmd in commands:
  q=subprocess.run(cmd,cwd=P,env=dict(os.environ,PYTHONPATH=str(P),PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
  results.append(dict(command=cmd,returncode=q.returncode,stdout=q.stdout,stderr=q.stderr));assert q.returncode==0,results[-1]
 for p,h in pins.items():assert sha(p)==h,p
 (R/'CPU-TESTS.json').write_text(json.dumps(results,indent=2)+'\n')
 ready=dict(status='CPU_READY_PENDING_REVIEW_GPU',pins=len(pins),pins_sha256=sha(R/'PINS.json'),tests_sha256=sha(R/'CPU-TESTS.json'),cpu_tests=13,dry_entries=2,isolated_gpu_pending=52,model_shadow_then_replace_pending=16,framework_import_gpu_query_compile_cache_actions=0)
 (R/'CPU-READY.json').write_text(json.dumps(ready,indent=2)+'\n');print(json.dumps(ready))
if __name__=='__main__':main()
