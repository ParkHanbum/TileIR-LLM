import unittest,ast
from pathlib import Path
from .run import result_gate
class Tests(unittest.TestCase):
 def test_sources(self):
  for p in [Path(__file__).parent/'run.py',Path(__file__).parent/'launch.py',Path(__file__).parent.parent/'tileir_gdn_wy_v1/kernels.py']:ast.parse(p.read_text())
 def test_gate(self):
  rows=[dict(arm=a,repeat=i,passed=True) for a in ('native','tile') for i in range(2)]
  r=dict(complete=True,records=rows,initial_flags=[True,True],final_flags=[True,True]);self.assertTrue(result_gate(r)['raw_pass']);rows[2]['passed']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_partial(self):self.assertFalse(result_gate(dict(complete=True,records=[]))['raw_pass'])
 def test_coverage(self):
  ids=[(seq*53+t)*4096+h*128+part*64+c for seq in range(32) for h in range(32) for part in range(2) for t in range(53) for c in range(64)]
  self.assertEqual(len(ids),6946816);self.assertEqual(len(set(ids)),6946816);self.assertEqual(max(ids),6946815)
 def test_source_contract(self):
  s=(Path(__file__).parent.parent/'tileir_gdn_wy_v1/kernels.py').read_text();self.assertIn('h//2',s);self.assertEqual(s.count('ct.mma('),2);self.assertIn('mask=t[:,None]<53',s)
if __name__=='__main__':unittest.main()
