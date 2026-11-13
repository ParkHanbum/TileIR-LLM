import ast,unittest
from pathlib import Path
from result_gate import result_gate
class Tests(unittest.TestCase):
 def fixture(self):return dict(complete=True,gpu_executed=True,records=[dict(phase=p,arm=a,mapped_owner_pinned=True,mapped_pointer_equal=True,repeats=[dict(call_count=1,full_storage_raw_equal=[True,True],input_unchanged=True,first_return_same_object=True,rejected_disjoint=True,output_layout_equal=True,repeat_raw_equal=True) for _ in range(2)]) for p in ('prefill','decode') for a in ('native','tileir')])
 def test_positive(self):self.assertTrue(result_gate(self.fixture())['raw_pass'])
 def test_missing_native(self):
  r=self.fixture();r['records'].pop();self.assertFalse(result_gate(r)['complete'])
 def test_failures(self):
  for k in ('input_unchanged','first_return_same_object','rejected_disjoint','output_layout_equal','repeat_raw_equal'):
   r=self.fixture();r['records'][0]['repeats'][0][k]=False;self.assertTrue(result_gate(r)['raw_difference_hold'])
 def test_early_hold(self):self.assertTrue(result_gate(dict(raw_difference_hold=True))['raw_difference_hold'])
 def test_ast_no_cuda(self):
  import sys
  self.assertNotIn('torch',sys.modules)
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
