"""CPU construction utility; no live binding changes or CUDA imports."""
import inspect,textwrap,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
def build(name,original,finite_all):
 m=json.loads((R/'MANIFEST.json').read_text());spec=next(x for x in m['bodies']if x['name']==name)
 assert hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()==spec['sha256']
 assert textwrap.dedent(inspect.getsource(original)).strip()==(R/'bodies'/(name+'.original.py.txt')).read_text().strip()
 env=dict(original.__globals__);env['_finite_all']=finite_all
 exec(compile('from __future__ import annotations\n'+(R/'bodies'/(name+'.candidate.py.txt')).read_text(),str(R/'bodies'/(name+'.candidate.py.txt')),'exec'),env)
 return env[spec['function']]
