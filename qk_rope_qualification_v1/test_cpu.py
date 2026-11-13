import unittest
from run import result_gate
class Tests(unittest.TestCase):
 def rows(self):return [dict(phase=p,branch=b,repeat=i,contract_pass=True,repeat_equal=True,comparisons={n:dict(raw_equal=True,candidate_nonfinite=0,reference_nonfinite=0) for n in ('q','k','gate')}) for p in ('prefill','decode') for b in ('native','tileir') for i in (0,1)]
 def test_all_required(self):self.assertTrue(result_gate(dict(complete=True,records=self.rows()))['raw_pass'])
 def test_mismatch(self):
  r=self.rows();r[2]['comparisons']['q']['raw_equal']=False;self.assertFalse(result_gate(dict(complete=True,records=r))['raw_pass'])
 def test_missing(self):self.assertFalse(result_gate(dict(complete=True,records=self.rows()[:-1]))['raw_pass'])
if __name__=='__main__':unittest.main()
