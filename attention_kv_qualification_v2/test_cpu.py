import unittest,ast,sys
from pathlib import Path
from .run import result_gate,selected_gate
class Tests(unittest.TestCase):
 def test_complete(self):
  rows=[dict(phase=p,arm=a,repeat=n,passed=True) for p in ('prefill','decode') for a in ('native','tile') for n in range(2)]
  self.assertTrue(result_gate({'complete':True,'records':rows})['raw_pass'])
  self.assertFalse(result_gate({'complete':True,'records':[rows[0]]*8})['raw_pass'])
 def test_partial_hold(self):self.assertFalse(result_gate({'complete':True,'records':[{'passed':True}]*7})['raw_pass'])
 def test_failure(self):self.assertTrue(result_gate({'raw_difference_hold':True})['raw_difference_hold'])
 def test_selected_metrics_gate(self):
  good=dict(raw_equal=True,reference_nonfinite=0,candidate_nonfinite=0,strict_failures=0)
  self.assertTrue(selected_gate(dict(key=good,value=good)))
  for field in ('reference_nonfinite','candidate_nonfinite','strict_failures'):
   bad=dict(good);bad[field]=1;self.assertFalse(selected_gate(dict(key=good,value=bad)))
  bad=dict(good);bad['raw_equal']=False;self.assertFalse(selected_gate(dict(key=bad,value=good)))
 def test_ast_no_gpu(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
  self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main()
