import ast,pathlib,types,unittest
from collections import Counter
R=pathlib.Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def classes(self):
  t=ast.parse((R/'adapter.py').read_text());ns={'Counter':Counter,'LIMITS':{'greedy.nan':16,'greedy.posinf':16,'greedy.rows':16,'ranks.nan':16,'ranks.posinf':16,'ranks.gather':16},'RANGE_LIMITS':{'greedy':16,'ranks':16,'scores':16}}
  exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,ast.ClassDef)and n.name in('Range','Sampling')],type_ignores=[]),'lean-callbacks','exec'),ns);return ns
 def test_counter_lifecycle(self):
  n=self.classes();c=types.SimpleNamespace(failed=None,active=True)
  for name,key in [('Range','RANGE_LIMITS'),('Sampling','LIMITS')]:
   x=n[name](c)
   with self.assertRaises(AssertionError):x.start('shadow')
   x.start('replace')
   with self.assertRaises(AssertionError):x.finish()
   x.counts=Counter(n[key])
   if name=='Sampling':x.temperature_count=16
   result=x.finish();self.assertFalse(x.active);self.assertEqual(result['calls'],sum(n[key].values()))
 def test_no_native_oracle_or_transfer(self):
  t=ast.parse((R/'adapter.py').read_text());bad={'cpu','equal','isnan','isposinf','isfinite','index_select','clone','synchronize','tolist','numpy'}
  self.assertFalse([n.attr for n in ast.walk(t)if isinstance(n,ast.Attribute)and n.attr in bad])
 def test_same_validated_sources(self):
  t=ast.parse((R/'adapter.py').read_text());text=(R/'adapter.py').read_text()
  self.assertIn('from sampling_guards144_runtime_v2 import adapter as proof',text)
  self.assertFalse(any(isinstance(n,ast.FunctionDef)and n.name in('apply','install_classes','prepare','guarded_body')for n in ast.walk(t)))
 def test_failed_controller_blocks(self):
  n=self.classes();c=types.SimpleNamespace(failed='latched',active=True)
  for k in ('Range','Sampling'):
   with self.assertRaises(AssertionError):n[k](c).start('replace')
if __name__=='__main__':unittest.main()
