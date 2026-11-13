import unittest
from .launch import result_gate
class Tests(unittest.TestCase):
 def good(self):return dict(complete=True,gpu_executed=True,records=[dict(phase=p,input_unchanged=True,reference_repeat_raw=True,finite=True,raw_equal=True) for p in ('prefill','decode')])
 def test_pass(self):self.assertTrue(result_gate(self.good())['raw_pass'])
 def test_raw_hold(self):
  r=self.good();r['records'][0]['raw_equal']=False;self.assertTrue(result_gate(r)['raw_difference_hold'])
 def test_incomplete(self):
  r=self.good();r['records'].pop();self.assertFalse(result_gate(r)['complete'])
 def test_input_changed(self):
  r=self.good();r['records'][0]['input_unchanged']=False;self.assertFalse(result_gate(r)['complete'])
if __name__=='__main__':unittest.main()
