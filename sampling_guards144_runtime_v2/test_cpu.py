import ast,json,unittest,types
from pathlib import Path
from collections import Counter
from .owned import prepare,R
class Tests(unittest.TestCase):
 def original(self,key):
  spec=json.loads((R/'SOURCE.json').read_text())[key];tree=ast.parse(Path(spec['path']).read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.lineno==spec['firstlineno'])
  env={'_finite_all':object(),'_range_guard':object(),'__name__':'source_fixture'}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),spec['path'],'exec'),env);return env[fn.name]
 def test_owned_callbacks(self):
  for key in ('greedy_api','ranks_api'):
   fn=self.original(key);cb=lambda *a:True
   body,proof=prepare(fn,key,cb,cb)
   self.assertIs(body.__globals__['_finite_all'],fn.__globals__['_finite_all'])
   self.assertIs(body.__globals__['_range_guard'],fn.__globals__['_range_guard'])
   self.assertIs(body.__globals__['_sampling_predicate'],cb)
 def test_wrong_source(self):
  fn=self.original('ranks_api');fn.__code__=fn.__code__.replace(co_filename='wrong')
  with self.assertRaises(AssertionError):prepare(fn,'ranks_api',None,None)
 def test_only_changed_expressions(self):
  for site,spec in json.loads((R/'SOURCE.json').read_text()).items():
   text=(R/(site+'.original.py.txt')).read_text()
   for a,b in spec['replacements']:self.assertEqual(text.count(a),1);text=text.replace(a,b)
   self.assertEqual(text,(R/(site+'.candidate.py.txt')).read_text())
 def test_nan_shortcircuit(self):
  tree=ast.parse((R/'greedy_api.candidate.py.txt').read_text());node=next(n for n in ast.walk(tree)if isinstance(n,ast.Assert)and 'greedy.nan' in ast.unparse(n));calls=[]
  def pred(site,*a):calls.append(site);return True
  with self.assertRaises(AssertionError):exec(compile(ast.Module(body=[node],type_ignores=[]),'short','exec'),{'_sampling_predicate':pred,'logits':None})
  self.assertEqual(calls,['greedy.nan'])
 def test_bounds_before_gather(self):
  fn=ast.parse((R/'ranks_api.candidate.py.txt').read_text()).body[0]
  start=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assert)and '_range_guard' in ast.unparse(n))
  nodes=fn.body[start:start+4];calls=[]
  env={'_range_guard':lambda *a:False,'_sampling_gather':lambda *a:calls.append('gather'),'_finite_all':lambda *a:True,'_sampling_predicate':lambda *a:False,'logits':None,'token_ids':None}
  with self.assertRaises(AssertionError):exec(compile(ast.Module(body=nodes,type_ignores=[]),'bound','exec'),env)
  self.assertEqual(calls,[])
 def test_selected_finite_before_nan(self):
  fn=ast.parse((R/'ranks_api.candidate.py.txt').read_text()).body[0]
  start=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assert)and '_range_guard' in ast.unparse(n));calls=[]
  env={'_range_guard':lambda *a:True,'_sampling_gather':lambda *a:calls.append('gather'),'_finite_all':lambda *a:False,'_sampling_predicate':lambda *a:calls.append('nan'),'logits':None,'token_ids':None}
  with self.assertRaises(AssertionError):exec(compile(ast.Module(body=fn.body[start:start+4],type_ignores=[]),'bound','exec'),env)
  self.assertEqual(calls,['gather'])
 def check_class(self):
  tree=ast.parse((R/'checks.py').read_text());env={'Counter':Counter}
  exec(compile(tree,'checks','exec'),env);return env['Checks'],env['LIMITS']
 def test_phase_and_census(self):
  C,limits=self.check_class();control=types.SimpleNamespace(active=True,failed=None)
  c=C(control)
  with self.assertRaises(AssertionError):c.start('replace')
  c.start('shadow');c.counts=Counter(limits);c.rows=[{'stage':'verified'}for _ in range(96)];c.temperatures=[{}]*16
  r=c.finish();self.assertTrue(r['shadow_pass']);self.assertEqual(r['calls'],96)
  c.start('replace');self.assertEqual(c.rows,[]);self.assertEqual(c.temperatures,[])
  with self.assertRaises(AssertionError):c.finish()
 def test_restore_order_and_conflict(self):
  tree=ast.parse((R/'adapter.py').read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Sampler')
  events=[]
  class Base:
   def close(self):events.append('range_close');return {'restored':True}
  env={'ranged':types.SimpleNamespace(Sampler=Base)}
  exec(compile(ast.Module(body=[cl],type_ignores=[]),'adapter','exec'),env);s=env['Sampler']()
  old,new=object(),object();mapping={'x':new};s.sampling_bindings=[(mapping,'x',old,new)]
  s.close();self.assertIs(mapping['x'],old);self.assertEqual(events,['range_close'])
  mapping['x']=object()
  with self.assertRaises(RuntimeError):s.close()
  self.assertEqual(events,['range_close','range_close'])
 def test_gate_rejects_missing_fields(self):
  tree=ast.parse((R/'gates.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='sample_gate');env={'LIMITS':self.check_class()[1]}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'gate','exec'),env)
  with self.assertRaises((AssertionError,KeyError)):env['sample_gate']({},'replace')
 def test_parse(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
