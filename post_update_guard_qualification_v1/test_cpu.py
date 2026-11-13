import unittest,copy
from .run import result_gate
from public_post_update_guard_v1.cases import cases
from public_post_update_guard_v1.contract import MESSAGES
class GateTests(unittest.TestCase):
 def good(self):
  rows=[]
  for label,d,error in cases():
   n=9 if error is None else(8 if label=='delta_overflow'else MESSAGES.index(error)+1)
   rows.append(dict(case=label,expected=error,native=[error]*2,candidate=[error]*2,passed=True,readonly_checks=4,stream_unchanged=True,candidate_stages=[list(range(n))]*2))
  return dict(complete=True,records=rows,final_synchronize=True)
 def test_good(self):self.assertTrue(result_gate(self.good())['raw_pass'])
 def test_incomplete(self):
  r=self.good();r['complete']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_later_stage_rejected(self):
  r=self.good();r['records'][1]['candidate_stages'][0].append(1);self.assertFalse(result_gate(r)['raw_pass'])
 def test_error_mismatch(self):
  r=self.good();r['records'][1]['candidate'][0]='other';self.assertFalse(result_gate(r)['raw_pass'])
 def test_missing_readonly(self):
  r=self.good();r['records'][0]['readonly_checks']=3;self.assertFalse(result_gate(r)['raw_pass'])
if __name__=='__main__':unittest.main()
