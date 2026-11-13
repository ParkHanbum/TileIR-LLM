"""Explicit full CPU freeze only after coordinator permits host hashing; never CUDA."""
import argparse,hashlib,json,os,pathlib,subprocess,sys
R=pathlib.Path(__file__).resolve().parent;P=R.parent;A=P/'public_sampling_guard_v1'
sys.path.insert(0,str(P))
PYTHON='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
 if not a.freeze:print('PENDING: full freeze requires coordinator host window');return
 assert sys.executable==PYTHON,'Use actual runtime venv'
 for name in ['MANIFEST.json','CPU-PINS.json','CPU-READY.json']:assert not (R/name).exists(),'Already frozen; preserve and version'
 forbidden={'__pycache__','cache_v1','tile','tiletmp','cuda','tmp'}
 assert not any(p.name in forbidden or p.suffix in ('.pyc','.cubin','.bytecode','.pt')for root in (R,A)for p in root.rglob('*'))
 manifest=json.loads((R/'MANIFEST.template.json').read_text());manifest.update(api=str(A/'api.py'),api_sha256=sha(A/'api.py'),kernel=str(A/'kernel.py'),kernel_sha256=sha(A/'kernel.py'))
 (R/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 from sampling_guard_synthetic_qualification_v1.prerequisite import verify,LINKS
 proof=verify()
 pins=json.loads((P/'i64_range_synthetic_qualification_v1/CPU-PINS.json').read_text())
 for rel,h in LINKS.values():pins[str(P/rel)]=h
 # Same reviewed launcher dependencies, reverified rather than copying any cache/receipt.
 for path,h in pins.items():assert sha(pathlib.Path(path))==h,('Inherited source drift',path)
 for root in (A,R):
  for f in root.iterdir():
   if f.is_file()and f.suffix in ('.py','.md','.json')and f.name not in ('CPU-PINS.json','CPU-READY.json'):pins[str(f)]=sha(f)
 (R/'CPU-PINS.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
 env=dict(os.environ,PYTHONPATH=str(P),PYTHONDONTWRITEBYTECODE='1')
 checks=[]
 for cmd in [[PYTHON,'-B','-m','unittest','public_sampling_guard_v1.test_cpu','sampling_guard_synthetic_qualification_v1.test_cpu'],[PYTHON,'-B',str(R/'run.py'),'--manifest',str(R/'MANIFEST.json'),'--manifest-sha',sha(R/'MANIFEST.json'),'--output',str(R/'DRY-NOT-CREATED')],[PYTHON,'-B',str(R/'launch.py'),'--manifest',str(R/'MANIFEST.json'),'--manifest-sha',sha(R/'MANIFEST.json'),'--run-name','samplingGuardSynthetic01']]:
  r=subprocess.run(cmd,cwd=P,env=env,capture_output=True,text=True,timeout=60);checks.append(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,checks[-1]
 assert not (R/'DRY-NOT-CREATED').exists()and not (R/'samplingGuardSynthetic01').exists()
 (R/'CPU-TESTS.json').write_text(json.dumps(checks,indent=2)+'\n')
 for path,h in pins.items():assert sha(pathlib.Path(path))==h
 ready=dict(status='READY_FOR_INDEPENDENT_REVIEW',pins=len(pins),pins_sha256=sha(R/'CPU-PINS.json'),manifest_sha256=sha(R/'MANIFEST.json'),tests_sha256=sha(R/'CPU-TESTS.json'),tests=11,dry_entries=2,actual_gpu_pending=True,model_binding_pending=True,source_only_no_cache_or_runtime=True,framework_gpu_query_compile_cache=False)
 (R/'CPU-READY.json').write_text(json.dumps(ready,indent=2)+'\n');print(json.dumps(ready))
if __name__=='__main__':main()
