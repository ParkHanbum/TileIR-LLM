import ast,unittest
from pathlib import Path
from run import result_gate,validate_manifest
class Tests(unittest.TestCase):
 def fixture(self):return dict(complete=True,gpu_executed=True,initial_flags=[True,True],final_flags=[True,True],records=[dict(arm=a,repeat=i,passed=True) for a in ('native','tile') for i in range(2)])
 def test_gate(self):self.assertTrue(result_gate(self.fixture())['raw_pass'])
 def test_incomplete(self):
  r=self.fixture();r['complete']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_order(self):
  r=self.fixture();r['records'].reverse();self.assertFalse(result_gate(r)['raw_pass'])
 def test_flags(self):
  r=self.fixture();r['final_flags']=[False,True];self.assertFalse(result_gate(r)['raw_pass'])
 def test_failure(self):
  r=self.fixture();r['records'][2]['passed']=False;r['raw_difference_hold']=True;self.assertTrue(result_gate(r)['raw_difference_hold']);self.assertFalse(result_gate(r)['raw_pass'])
 def test_bad_manifest(self):
  with self.assertRaises(AssertionError):validate_manifest(dict(schema='wrong'))
 def test_json_terminator(self):
  tree=ast.parse((Path(__file__).parent/'run.py').read_text())
  save=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='save')
  self.assertEqual(save.body[0].value.args[0].right.value,'\n')
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
