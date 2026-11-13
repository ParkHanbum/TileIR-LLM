import unittest,copy
from .modules_gdn import exact_bytes,audit,dependencies,EXPECTED_IDENTITIES
class Tests(unittest.TestCase):
 def test_manifest_available(self):self.assertGreater(len(dependencies()),1)
 def test_cpu_copy_corruption_detector(self):
  original=bytes(range(64));copied=bytearray(original);exact_bytes(original,bytes(copied))
  for offset in (0,31,63):
   bad=bytearray(copied);bad[offset]^=1
   with self.assertRaises(AssertionError):exact_bytes(original,bytes(bad))
  self.assertEqual(bytes(copied),original)
 def test_state_and_contract_gate(self):
  r=dict(complete=True,records=[dict(bank=bank,backend=backend,same_stream=True,nonalias=True,passed=True,contract=True,output_raw=True,state_raw=True,finite=True,output_corruption_detected=True,state_corruption_detected=True)for bank,backend in EXPECTED_IDENTITIES])
  self.assertTrue(audit(r))
  for field in ('passed','contract','output_raw','state_raw','finite','output_corruption_detected','state_corruption_detected','same_stream','nonalias'):
   bad=copy.deepcopy(r);bad['records'][1][field]=False;self.assertFalse(audit(bad))
  for change in ('duplicate','missing','wrongbank','reverse'):
   bad=copy.deepcopy(r)
   if change=='duplicate':bad['records'][1]=copy.deepcopy(bad['records'][0])
   elif change=='missing':bad['records'].pop()
   elif change=='wrongbank':bad['records'][0]['bank']='invented'
   else:bad['records'].reverse()
   self.assertFalse(audit(bad))
if __name__=='__main__':unittest.main()
