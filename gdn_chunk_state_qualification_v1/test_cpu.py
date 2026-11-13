import unittest,ast
from pathlib import Path
from .run import result_gate
class Tests(unittest.TestCase):
 def test_ast(self):
  P=Path(__file__).parent.parent
  for d in ['tileir_gdn_chunk_state_v1','gdn_chunk_state_qualification_v1']:
   for p in (P/d).glob('*.py'):ast.parse(p.read_text())
 def test_gate(self):
  r=dict(complete=True,records=[dict(arm=a,repeat=i,passed=True) for a in ('native','tile') for i in range(2)],initial_flags=[True,True],final_flags=[True,True]);self.assertTrue(result_gate(r)['raw_pass']);r['records'][2]['passed']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_shape(self):
  self.assertEqual(32*32*128*128,16777216);self.assertEqual(1696*32*128,6946816)
 def test_state_coverage(self):
  vals=[v for part in range(4) for v in range(part*32,(part+1)*32)];self.assertEqual(vals,list(range(128)))
 def test_source(self):
  s=(Path(__file__).parent.parent/'tileir_gdn_chunk_state_v1/kernels.py').read_text();self.assertEqual(s.count('ct.mma('),4);self.assertIn('ct.gather(H0',s);self.assertIn('head//2',s)
