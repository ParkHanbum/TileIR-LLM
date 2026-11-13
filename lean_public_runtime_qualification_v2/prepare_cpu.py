"""Run only after coordinator permits actual CPU runtime regression. No GPU."""
import ast,hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (R/'PINS.json').exists(),'Do not silently refreeze'
assert not (R/'cache_v1').exists() and not (R/'CACHE-DESTINATION-RECEIPT.json').exists()
parents=[P/'lean_public_runtime_qualification_v1/PINS.json',P/'all_gemm_expanded_composition_v4/PINS.json',P/'lean_public_composition_v2/CPU-PINS.json']
pins={}
for f in parents:
 for p,h in json.loads(f.read_text()).items():
  assert p not in pins or pins[p]==h,('Conflicting inherited source',p)
  assert sha(p)==h,('Inherited source changed',p)
  pins[p]=h
 pins[str(f)]=sha(f)
for folder in (R,P/'lean_public_composition_v2'):
 for f in folder.iterdir():
  if f.suffix in ('.py','.md','.json') and f.name!='PINS.json':pins[str(f)]=sha(f)
for ref in json.loads((R/'PREREQUISITE.json').read_text()).values():
 if isinstance(ref,dict):
  assert sha(ref['path'])==ref['sha256'];pins[ref['path']]=ref['sha256']
for f in R.glob('*.py'):ast.parse(f.read_text())
cmd=[sys.executable,'-B','-m','unittest','lean_public_runtime_qualification_v2.test_cpu','lean_public_composition_v2.test_sampler_cpu','lean_public_composition_v2.test_composition_cpu','lean_public_composition_v2.test_attention_runner_cpu','lean_public_composition_v2.test_core_parent_cpu']
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(P),'CUDA_VISIBLE_DEVICES':''}
p=subprocess.run(cmd,cwd=P,env=env,capture_output=True,timeout=90)
(R/'cpu.stdout').write_bytes(p.stdout);(R/'cpu.stderr').write_bytes(p.stderr)
assert p.returncode==0,p.stderr.decode()
assert all(sha(p)==h for p,h in pins.items())
(R/'CPU-TESTS.json').write_text(json.dumps(dict(actual_exit=p.returncode,command=cmd,stdout_sha256=sha(R/'cpu.stdout'),stderr_sha256=sha(R/'cpu.stderr'),gpu=False),indent=2)+'\n')
for name in ('CPU-TESTS.json','cpu.stdout','cpu.stderr'):pins[str(R/name)]=sha(R/name)
(R/'PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
(R/'CPU-READY.json').write_text(json.dumps(dict(cpu_ready=True,gpu_release=False,prerequisite_complete=True,pins_count=len(pins),pins_sha256=sha(R/'PINS.json'),independent_review_required=True),indent=2)+'\n')
print(len(pins),sha(R/'PINS.json'))
