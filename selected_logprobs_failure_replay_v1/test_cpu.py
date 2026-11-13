import unittest
from run import result_gate
class Gates(unittest.TestCase):
 def base(self):return dict(complete=True,records=[dict(arm=a,repeat=n,binding_pass=True,vs_native={'raw_equal':a=='native'}) for a in ('native','tile') for n in (0,1)])
 def test_reproduced_difference_not_success(self):
  g=result_gate(self.base());self.assertTrue(g['contract_pass']);self.assertTrue(g['raw_difference_hold']);self.assertFalse(g['raw_pass'])
 def test_order(self):
  r=self.base();r['records'][2]=r['records'][0];self.assertFalse(result_gate(r)['complete'])
 def test_binding_failure(self):
  r=self.base();r['records'][0]['binding_pass']=False;self.assertFalse(result_gate(r)['contract_pass'])
 def test_incomplete(self):
  r=self.base();r['complete']=False;self.assertFalse(result_gate(r)['raw_pass'])
if __name__=='__main__':unittest.main()
