import unittest,ast,json
from pathlib import Path
from .run import validate_manifest,result_gate
R=Path(__file__).parent;P=R.parent
class Tests(unittest.TestCase):
 def test_single_numerical_delta(self):
  old=(P/'tileir_paged_attention_v7/kernels.py').read_text();new=(P/'tileir_paged_attention_v8/kernels.py').read_text()
  changed=' inv_den=ct.truediv(1.0,den,rounding_mode=ct.RoundingMode.RN)\n result=ct.mul(ct.reshape(ct.extract(pv,(0,0),(1,256)),(256,)),inv_den,rounding_mode=ct.RoundingMode.RN)'
  self.assertEqual(new.replace(changed,' result=ct.reshape(ct.extract(pv,(0,0),(1,256)),(256,))/den'),old)
 def test_api_unchanged(self):
  for n in ('api.py','contract.py','__init__.py'):self.assertEqual((P/'tileir_paged_attention_v7'/n).read_bytes(),(P/'tileir_paged_attention_v8'/n).read_bytes())
 def test_bank_order(self):
  s=validate_manifest(json.loads((R/'MANIFEST.json').read_text()));self.assertEqual([x['q']['metadata']['shape'][0] for x in s],[1696,1696,32]);self.assertIn('.layers.7.',s[0]['module'])
 def test_bad_order(self):
  m=json.loads((R/'MANIFEST.json').read_text());m['banks'].reverse()
  with self.assertRaises(AssertionError):validate_manifest(m)
 def test_result_gate(self):
  r=dict(complete=True,records=[dict(passed=True) for _ in range(12)]);self.assertTrue(result_gate(r)['raw_pass']);r['records'][2]['passed']=False;self.assertFalse(result_gate(r)['raw_pass']);self.assertFalse(result_gate(dict(complete=True,records=[]))['raw_pass'])
 def test_global_failstop(self):
  s=(R/'run.py').read_text();self.assertIn("('native','native','tile','tile')",s);self.assertIn("r['raw_difference_hold']=True;save();return 2",s);self.assertIn('extent=checked_extent(kd,vd)',s)
 def test_ast(self):
  for d in (R,P/'tileir_paged_attention_v8'):
   for f in d.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
