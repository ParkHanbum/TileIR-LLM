import unittest,copy
from gdn_decode_core_qualification_v4.storage_contract import untouched_ranges
from gdn_decode_core_qualification_v4.run import result_gate
class Tests(unittest.TestCase):
 def test_ranges(self):
  m=dict(shape=[4,1,2,2],stride=[7,4,2,1],storage_offset=3)
  self.assertEqual(untouched_ranges(m,[2,0]),[(0,3),(7,17),(21,28)])
 def test_duplicate(self):
  with self.assertRaises(AssertionError):untouched_ranges(dict(shape=[4,1,2,2],stride=[7,4,2,1],storage_offset=3),[1,1])
 def test_gates(self):
  v=dict(raw_equal=True,strict_pass=True,candidate_nonfinite=0,reference_nonfinite=0)
  r=dict(complete=True,records=[dict(contract_pass=True,comparisons=dict(output=v,state=v))for _ in range(2)])
  self.assertTrue(result_gate(r)['raw_pass']);r=copy.deepcopy(r)
  r['records'][1]['comparisons']['state']=dict(v,raw_equal=False)
  self.assertFalse(result_gate(r)['raw_pass']);self.assertTrue(result_gate(r)['raw_difference_hold'])
  r['records'][1]['contract_pass']=False;self.assertFalse(result_gate(r)['raw_difference_hold'])
if __name__=='__main__':unittest.main()
