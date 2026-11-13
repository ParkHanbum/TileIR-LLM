import ast,unittest
from pathlib import Path
from result_gate import result_gate
class Tests(unittest.TestCase):
 def fixture(self):return dict(complete=True,gpu_executed=True,records=[dict(phase=p,repeats=[dict(full_storage_raw_equal=True,input_unchanged=True,views_alias_expected=True,full_capacity_pad=True,repeat_raw_equal=True) for _ in range(2)]) for p in ('prefill','decode')])
 def test_positive(self):self.assertTrue(result_gate(self.fixture())['raw_pass'])
 def test_missing(self):
  r=self.fixture();r['records'].pop();self.assertFalse(result_gate(r)['complete'])
 def test_each_failure(self):
  for k in ('full_storage_raw_equal','input_unchanged','views_alias_expected','full_capacity_pad','repeat_raw_equal'):
   r=self.fixture();r['records'][0]['repeats'][0][k]=False;self.assertTrue(result_gate(r)['raw_difference_hold'])
 def test_ast_cpu(self):
  import sys
  self.assertNotIn('torch',sys.modules)
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
