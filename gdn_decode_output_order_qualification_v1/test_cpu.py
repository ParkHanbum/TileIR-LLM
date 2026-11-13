import unittest,json
from pathlib import Path
from gdn_decode_output_order_qualification_v1.run import result_gate
P=Path(__file__).resolve().parent.parent
class Tests(unittest.TestCase):
 def test_oneline(self):
  a=(P/'tileir_gdn_decode_core_v3/kernels.py').read_text();b=(P/'tileir_gdn_decode_core_v4/kernels.py').read_text();self.assertEqual(a.replace('total=h1*ct.reshape(q1,(1,32))+h0*ct.reshape(q0,(1,32))','total=h0*ct.reshape(q0,(1,32))+h1*ct.reshape(q1,(1,32))'),b)
 def test_gate(self):
  r=dict(complete=True,records=[dict(contract_pass=True,passed=True)for _ in range(6)]);self.assertTrue(result_gate(r)['raw_pass']);r['records'][5]['passed']=False;self.assertFalse(result_gate(r)['raw_pass']);r['records'].pop();self.assertFalse(result_gate(r)['complete'])
if __name__=='__main__':unittest.main()
