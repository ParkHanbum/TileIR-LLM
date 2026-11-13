"""CPU freeze after coordinator GO only; no GPU/model/cache operation."""
import ast,hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (R/'PINS.json').exists() and not (R/'cache_v1').exists()
pins=json.loads((P/'lean_public_runtime_qualification_v2/PINS.json').read_text())
for p,h in pins.items():assert sha(p)==h,p
for name in ('HELPER-SPECS.json','SETUP-SPECS.json'):
 spec=json.loads((R/name).read_text());assert spec['complete']
 for h in spec['helpers']:
  assert sha(h['source_path'])==h['source_sha256'];pins[h['source_path']]=h['source_sha256']
  tree=ast.parse(Path(h['source_path']).read_text());matches=[n for n in ast.walk(tree)if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name==h['attribute']]
  assert any([a.arg for a in n.args.posonlyargs+n.args.args+n.args.kwonlyargs]==h['parameter_names']for n in matches),(h['label'],'public AST signature mismatch')
 for f in spec.get('source_specs',[]):pins[f]=sha(f)
for f in (P/'lean_public_runtime_qualification_v2/final_proof_01').iterdir():
 if f.is_file():pins[str(f)]=sha(f)
for f in R.iterdir():
 if f.suffix in ('.py','.json','.md'):pins[str(f)]=sha(f)
for f in R.glob('*.py'):ast.parse(f.read_text())
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(P),'CUDA_VISIBLE_DEVICES':''}
commands=[[sys.executable,'-B','-m','unittest','lean25_metadata_inventory_v2.test_cpu','lean25_metadata_inventory_v2.test_lifetime'],[sys.executable,'-B',str(R/'run.py')],[sys.executable,'-B',str(R/'launch.py')]]
rows=[]
for i,cmd in enumerate(commands):
 p=subprocess.run(cmd,cwd=P,env=env,capture_output=True,timeout=90)
 for suffix,data in [('stdout',p.stdout),('stderr',p.stderr)]:
  f=R/f'cpu{i}.{suffix}';f.write_bytes(data);pins[str(f)]=sha(f)
 rows.append(dict(command=cmd,exit=p.returncode));assert p.returncode==0,p.stderr.decode()
assert all(sha(p)==h for p,h in pins.items())
(R/'CPU-TESTS.json').write_text(json.dumps(dict(rows=rows,tests=17,complete=True,gpu=False),indent=2)+'\n');pins[str(R/'CPU-TESTS.json')]=sha(R/'CPU-TESTS.json')
(R/'PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
(R/'CPU-READY.json').write_text(json.dumps(dict(cpu_ready=True,independent_review_required=True,gpu_release=False,pins=len(pins),pins_sha256=sha(R/'PINS.json'),cache_created=False,scope='One native reference plus observed lean metadata request; setup separate; no timing'),indent=2)+'\n')
print(len(pins),sha(R/'PINS.json'))
