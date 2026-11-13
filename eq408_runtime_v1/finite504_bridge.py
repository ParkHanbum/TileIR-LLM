"""Future composition only; standalone EQ408 does not activate this bridge."""
import inspect,textwrap,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent.parent

def attention_method(original,finite,eq):
 r=P/'floating_guard_504_design_v1';spec=json.loads((r/'SOURCE.json').read_text())['attention']
 source=textwrap.dedent(inspect.getsource(original))
 assert source.strip()==(r/'attention.original.py.txt').read_text().strip()
 candidate=(r/'attention.candidate.py.txt').read_text()
 assert candidate==source.replace(spec['old'],spec['new'])
 old='bool((t==1).all())';assert candidate.count(old)==1
 candidate=candidate.replace(old,'_eq_all(t,1)')
 ns=dict(original.__globals__);ns['_finite']=finite;ns['_eq_all']=eq
 exec(compile(candidate,'<explicit-finite504-eq408-attention>','exec'),ns)
 return ns[spec['method']]
