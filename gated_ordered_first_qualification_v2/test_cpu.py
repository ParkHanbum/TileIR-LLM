import ast,unittest
from pathlib import Path
class Test(unittest.TestCase):
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
 def test_binding_layout(self):
  s=(Path(__file__).parent/'run.py').read_text()
  for v in ('8192','12288','helper_vs_saved_native','clean_vs_saved_tile','instrumented_vs_clean','extra_compiledkernel_replay_launches=0'):self.assertIn(v,s)

class GateRegression(unittest.TestCase):
 def test_reject_old_tile_match_native_mismatch(self):
  import importlib.util
  p=Path(__file__).parent/'run.py';spec=importlib.util.spec_from_file_location('ordered_gate_under_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  r=dict(complete=True,inputs_preserved=True,rstd={'all_finite':True},helper_vs_saved_native={'raw_equal':True},clean_vs_saved_tile={'raw_equal':True},instrumented_vs_clean={'raw_equal':True},clean_vs_native={'raw_equal':False})
  self.assertFalse(m.result_gate(r)['raw_pass']);self.assertTrue(m.result_gate(r)['raw_difference_hold'])
  r['clean_vs_native']['raw_equal']=True;r['clean_vs_saved_tile']['raw_equal']=False
  self.assertTrue(m.result_gate(r)['raw_pass'])
