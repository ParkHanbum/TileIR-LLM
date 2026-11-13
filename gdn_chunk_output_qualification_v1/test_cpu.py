import unittest,ast
from pathlib import Path
from .run import result_gate
class Tests(unittest.TestCase):
 def test_ast(self):
  P=Path(__file__).parent.parent
  for d in ['tileir_gdn_chunk_output_v1','gdn_chunk_output_qualification_v1']:
   for p in (P/d).glob('*.py'):ast.parse(p.read_text())
 def test_gate(self):
  r=dict(complete=True,records=[dict(arm=a,repeat=i,passed=True) for a in ('native','tile') for i in range(2)],initial_flags=[True,True],final_flags=[True,True]);self.assertTrue(result_gate(r)['raw_pass']);r['records'][3]['passed']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_mask(self):
  pairs=[(a,b) for a in range(64) for b in range(64) if a>=b and a<53 and b<53];self.assertEqual(len(pairs),53*54//2)
 def test_address(self):
  self.assertEqual([x for p in range(4) for x in range(p*32,(p+1)*32)],list(range(128)));self.assertEqual(1696*32*128,6946816)
 def test_scale(self):self.assertEqual(0.08838834764831845,128**-0.5)
