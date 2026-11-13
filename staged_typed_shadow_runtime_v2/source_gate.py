import ast,inspect,textwrap,hashlib,json
from pathlib import Path
def fingerprint(source):
 return hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(source)),include_attributes=False).encode()).hexdigest()
def verify(b,bt,root):
 spec=json.loads((Path(root)/'METHODS.json').read_text());out=[]
 for row in spec:
  cls=getattr(b if row['module']=='buffer'else bt,row['owner']);fn=getattr(cls,row['attribute'])
  source,start=inspect.getsourcelines(fn)
  assert str(Path(inspect.getsourcefile(fn)).resolve())==row['path'] and start==row['lineno']
  assert fingerprint(''.join(source))==row['ast_sha256']
  assert list(inspect.signature(fn).parameters)==row['parameters']
  out.append(dict(owner=row['owner'],attribute=row['attribute'],source_verified=True))
 return out
