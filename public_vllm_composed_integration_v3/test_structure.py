import unittest
from structure import EXPECTED,TOTAL,require_full_evidence
class Tests(unittest.TestCase):
 def test_total(self):self.assertEqual(TOTAL,1616)
 def test_missing_sampler_blocks(self):
  with self.assertRaises(ValueError):require_full_evidence({k:{} for k in EXPECTED if k!='sample_counts'})
 def test_representative_not_full(self):
  d={k:dict(logical_calls=n,actual_exit=0,full_integration_pass=True,independent_recovery=True) for k,n in EXPECTED.items()};self.assertTrue(require_full_evidence(d));d['sample_counts']['logical_calls']=2
  with self.assertRaises(ValueError):require_full_evidence(d)
if __name__=='__main__':unittest.main()
