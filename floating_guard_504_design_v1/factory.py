"""Fresh method globals; no module patch and no automatic installation."""
import inspect,textwrap,json
from pathlib import Path
R=Path(__file__).parent

def copied_method(tag,original,finite):
 spec=json.loads((R/'SOURCE.json').read_text())[tag]
 source=textwrap.dedent(inspect.getsource(original))
 assert source.strip()==(R/(tag+'.original.py.txt')).read_text().strip()
 candidate=(R/(tag+'.candidate.py.txt')).read_text()
 assert candidate==source.replace(spec['old'],spec['new'])
 ns=dict(original.__globals__);ns['_finite']=finite
 exec(compile(candidate,str(R/(tag+'.candidate.py.txt')),'exec'),ns)
 return ns[spec['method']]

def contiguous_finite(site,x):
 # KV value is intentionally unsupported here, not silently materialized/cast.
 assert site!='kv.value','Strided KV value requires a separately qualified predicate'
 from tileir_validation_primitives_v3.api import all_finite
 return all_finite(x)
