import ast,copy,unittest
from pathlib import Path
from result_gate import result_gate
class Tests(unittest.TestCase):
 def fixture(self):
  return dict(complete=True,gpu_executed=True,records=[dict(phase=p,placement=d,mapped_owner_pinned=True,mapped_pointer_equal=True,repeats=[dict(all_group_storage_raw_equal=[True]*4,input_unchanged=True,views_alias_expected=True,repeat_raw_equal=True) for _ in range(2)]) for p in ('prefill','decode') for d in ('mapped_uva_primary','device_copy_control')])
 def test_positive(self):self.assertTrue(result_gate(self.fixture())['raw_pass'])
 def test_missing_control(self):
  r=self.fixture();r['records'].pop();self.assertFalse(result_gate(r)['complete'])
 def test_failure_flags(self):
  for field in ('input_unchanged','views_alias_expected','repeat_raw_equal'):
   r=self.fixture();r['records'][0]['repeats'][0][field]=False;self.assertTrue(result_gate(r)['raw_difference_hold'])
  r=self.fixture();r['records'][0]['repeats'][0]['all_group_storage_raw_equal'][3]=False;self.assertTrue(result_gate(r)['raw_difference_hold'])
 def test_uva_required(self):
  r=self.fixture();r['records'][0]['mapped_pointer_equal']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_ast_no_gpu_import(self):
  import sys
  self.assertNotIn('torch',sys.modules)
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
