import unittest,ast
from pathlib import Path
from .run import result_gate
class Tests(unittest.TestCase):
 def test_source(self):
  p=Path(__file__).parent.parent
  for d in ('tileir_gdn_solve_tril_v1','gdn_solve_tril_qualification_v1'):
   for f in (p/d).glob('*.py'):ast.parse(f.read_text())
 def test_partial(self):self.assertFalse(result_gate({'complete':False,'records':[]})['raw_pass'])
 def test_hold(self):self.assertTrue(result_gate({'raw_difference_hold':True})['raw_difference_hold'])
 def test_false_rows(self):
  r={'complete':True,'records':[dict(arm=a,repeat=i,passed=False) for a in ('native','tile') for i in range(2)],'initial_flags':[True,True],'final_flags':[True,True]};self.assertFalse(result_gate(r)['raw_pass'])
 def test_shape_algebra(self):
  self.assertEqual([len(range(2,min(16,53-n*16))) for n in range(4)],[14,14,14,3])
 def test_candidate_boundaries(self):
  s=(Path(__file__).parent.parent/'tileir_gdn_solve_tril_v1/kernels.py').read_text();self.assertNotIn('tfloat32',s);self.assertEqual(s.count('astype(ct.bfloat16'),1);self.assertEqual(s.count('for row in range'),4)
