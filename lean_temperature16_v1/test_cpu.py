import ast,pathlib,types,unittest
R=pathlib.Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def objects(self):
  t=ast.parse((R/'adapter.py').read_text());e={};exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))],type_ignores=[]),'lean-temp','exec'),e);return e
 def test_census_latch(self):
  e=self.objects();c=types.SimpleNamespace(failed=None);x=e['Checks'](c);x.start()
  with self.assertRaises(AssertionError):x.finish()
  x.calls=16;d=x.finish();self.assertEqual(d['native_predicates'],0);self.assertEqual(d['extra_checked_range_calls'],16)
  c.failed='latched'
  with self.assertRaises(AssertionError):x.start()
 def test_explicit_receipt(self):
  e=self.objects();s={'calls':96,'temperature_original_retained':16};t={'calls':16,'native_predicates':0};e['annotate_sampling_receipt'](s,t)
  self.assertEqual(s['temperature_original_retained'],0);self.assertEqual(s['temperature_observation_slots'],16)
  with self.assertRaises(AssertionError):e['annotate_sampling_receipt'](s,t)
 def test_no_diagnostics(self):
  t=ast.parse((R/'adapter.py').read_text());bad={'cpu','equal','index_select','clone','numpy','tolist','synchronize'}
  self.assertFalse([n.attr for n in ast.walk(t)if isinstance(n,ast.Attribute)and n.attr in bad])
if __name__=='__main__':unittest.main()
