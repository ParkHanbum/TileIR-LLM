import ast,inspect,textwrap,json,hashlib
from pathlib import Path
R=Path(__file__).parent
def prepare(original,tag,predicate):
 spec=json.loads((R/'SOURCES.json').read_text())[tag]
 assert inspect.getsourcefile(original)==spec['path'] and original.__code__.co_firstlineno==spec['firstlineno']
 assert hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()==spec['sha256']
 source=textwrap.dedent(inspect.getsource(original))
 assert ast.dump(ast.parse(source),include_attributes=False)==spec['ast']
 ns=dict(original.__globals__);ns['_eq_all']=predicate
 exec(compile('from __future__ import annotations\n'+(R/(tag+'.copied.py.txt')).read_text(),str(R/(tag+'.copied.py.txt')),'exec'),ns)
 return ns['apply']
