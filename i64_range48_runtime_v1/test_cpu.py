import ast,copy,json,unittest
from pathlib import Path
from .owned import prepare,R
class Tests(unittest.TestCase):
 def original(self,key):
  spec=json.loads((R/'SOURCE.json').read_text())[key];tree=ast.parse(Path(spec['path']).read_text(),filename=spec['path'])
  if key=='greedy':tree=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Greedy');fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='apply')
  else:fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
  ns={'__name__':'cpu_fixture','_finite_all':object()};exec(compile(ast.Module(body=[fn],type_ignores=[]),spec['path'],'exec'),ns);return ns[fn.name]
 def test_exact_runtime_source_path(self):
  for key in ('ranks','scores','greedy'):
   fn=self.original(key);body,proof=prepare(fn,key,lambda *a:True)
   self.assertEqual(proof['input_source_sha256'],json.loads((R/'SOURCE.json').read_text())[key]['input_sha256'])
   self.assertIs(body.__globals__['_finite_all'],fn.__globals__['_finite_all'])
 def test_wrong_filename(self):
  fn=self.original('ranks');fn.__code__=fn.__code__.replace(co_filename='wrong')
  with self.assertRaises(AssertionError):prepare(fn,'ranks',None)
 def test_only_expression_changed(self):
  for key,spec in json.loads((R/'SOURCE.json').read_text()).items():
   original=(R/(key+'.original.py.txt')).read_text();candidate=(R/(key+'.candidate.py.txt')).read_text()
   self.assertEqual(original.count(spec['expression']),1);self.assertEqual(candidate,original.replace(spec['expression'],spec['replacement']))
 def test_scores_shortcircuit(self):
  tree=ast.parse((R/'scores.candidate.py.txt').read_text());node=next(n for n in ast.walk(tree)if isinstance(n,ast.Assert)and isinstance(n.test,ast.BoolOp)and '_finite_all' in ast.unparse(n));calls=[]
  env={'_finite_all':lambda *a:False,'logits':None,'token_ids':None,'_range_guard':lambda *a:calls.append(a)}
  with self.assertRaises(AssertionError):exec(compile(ast.Module(body=[node],type_ignores=[]),'shortcircuit','exec'),env)
  self.assertEqual(calls,[])
 def test_sources_parse(self):
  for f in R.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
