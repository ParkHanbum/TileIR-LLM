import unittest
from adapter import remap_slots,extent
from result_gate import audit
class Tests(unittest.TestCase):
 def test_actual(self):
  s=list(range(4,129,4));selected,m=remap_slots(s,311);self.assertEqual(selected,s);self.assertEqual(m,list(range(1,33)))
 def test_null_and_permuted(self):
  s=list(range(4,129,4));s.reverse();s[0]=0;selected,m=remap_slots(s,311)
  self.assertEqual(m[0],0)
  for original,mapped in zip(s[1:],m[1:]):self.assertEqual(selected[mapped-1],original)
 def test_duplicates(self):
  with self.assertRaises(AssertionError):remap_slots([1]*32,311)
 def test_oob(self):
  with self.assertRaises(AssertionError):remap_slots(list(range(32)),31)
 def test_scratch_bound(self):self.assertLess(extent((33,8192,3),(1081344,1,8192))*2,80*1024*1024)
 def fixture(self):
  rows=[dict(passed=True,original_calls=1) for _ in range(360)];d=dict(shadow_pass=True,supported_calls=360,supported_vendor_fallback_calls=0,rows=rows,counts={'x:prefill':24,'x:decode':360})
  import copy
  b=copy.deepcopy(d)
  for row in b['rows']:row['original_calls']=0
  return dict(complete=True,engine_shutdown=True,adapter_restore=[{'restored':True}],public_logprob_gate={'passed':True},precision_policy={'restored':True,'restored_split_k':True,'changes':[]},shadow_request={'output_token_ids':[[1]*16]*32,'selected_dispatch':d},replacement_request={'output_token_ids':[[1]*16]*32,'selected_dispatch':b})
 def test_gate_positive(self):self.assertTrue(audit(self.fixture())['passed'])
 def test_badrow_hold(self):
  r=self.fixture();r['shadow_request']['selected_dispatch']['rows'][359]['passed']=False;self.assertFalse(audit(r)['passed'])
 def test_fallback_hold(self):
  r=self.fixture();r['replacement_request']['selected_dispatch']['supported_vendor_fallback_calls']=1;self.assertFalse(audit(r)['passed'])
if __name__=='__main__':unittest.main()
