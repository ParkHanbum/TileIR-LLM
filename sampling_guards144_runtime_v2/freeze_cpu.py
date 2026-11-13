"""One-time CPU freeze after coordinator host-window permission; no GPU."""
import argparse,hashlib,json,os,pathlib,subprocess,sys,ast
R=pathlib.Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
PY='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
def sha(p):
 with pathlib.Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--freeze',action='store_true');args=a.parse_args()
 if not args.freeze:print('PENDING host window');return
 assert sys.executable==PY
 for name in ['PINS.json','CPU-READY.json','cache_v1','CACHE-PRESEED-01.json']:assert not(R/name).exists()
 pins={}
 for folder,h in [('i64_range48_runtime_v1','4fa89fddcac33903e531c44724387de91ea74e8f66232f23667b2a952dea6756'),('sampling_guard_synthetic_qualification_v1','0dd3d19d8aaa5d4e0cbcf2bb42afc76a5da1e02b1d75e5fe4a289d5697e19015')]:
  p=P/folder/('PINS.json'if folder.startswith('i64_range48')else'CPU-PINS.json')
  assert sha(p)==h
  for path,digest in json.loads(p.read_text()).items():
   assert sha(path)==digest
   assert path not in pins or pins[path]==digest
   pins[path]=digest
  pins[str(p)]=h
 refs=json.loads((R/'PREREQUISITE.json').read_text())
 for ds in refs.values():
  for v in ds.values():assert sha(v['path'])==v['sha256'];pins[v['path']]=v['sha256']
 for f in R.iterdir():
  assert not f.is_symlink()
  if f.is_file()and f.suffix in('.py','.txt','.md','.json'):
   if f.suffix=='.py':ast.parse(f.read_text())
   pins[str(f)]=sha(f)
 (R/'PINS.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
 env=dict(os.environ,PYTHONPATH=str(P),PYTHONDONTWRITEBYTECODE='1')
 cmds=[[PY,'-B','-m','unittest','sampling_guards144_runtime_v2.test_cpu','sampling_guards144_runtime_v2.test_install_chain'],[PY,'-B',str(R/'run.py')],[PY,'-B',str(R/'launch.py')],[PY,'-B','-c','from sampling_guards144_runtime_v2.gates import qualification;from pathlib import Path;print(qualification(Path('+repr(str(R))+')))']]
 records=[]
 for cmd in cmds:
  p=subprocess.run(cmd,cwd=P,env=env,capture_output=True,text=True,timeout=90)
  records.append(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,records[-1]
 for path,h in pins.items():assert sha(path)==h
 (R/'CPU-TESTS.json').write_text(json.dumps(records,indent=2)+'\n')
 ready=dict(status='CPU_READY_PENDING_INDEPENDENT_REVIEW_AND_GPU',pins=len(pins),pins_sha256=sha(R/'PINS.json'),tests_sha256=sha(R/'CPU-TESTS.json'),tests=14,dry_entries=2,actual_prerequisite_checked=True,joint_boundaries=144,range48_model_prerequisite_assumed=False,extra_internal_range_checks=16,temperature_predicate_replaced=False,framework_gpu_query_compile_cache_actions=0)
 (R/'CPU-READY.json').write_text(json.dumps(ready,indent=2)+'\n');print(json.dumps(ready))
if __name__=='__main__':main()
