import unittest,ast
from pathlib import Path
from .run import result_gate
class Tests(unittest.TestCase):
 def test_ast(self):
  P=Path(__file__).parent.parent
  for d in ['tileir_gdn_chunk_output_v2','gdn_chunk_output_qualification_v2']:
   for p in (P/d).glob('*.py'):ast.parse(p.read_text())
 def test_gate(self):
  r=dict(complete=True,records=[dict(arm=a,repeat=i,passed=True) for a in ('native','tile') for i in range(2)],initial_flags=[True,True],final_flags=[True,True]);self.assertTrue(result_gate(r)['raw_pass']);r['records'][3]['passed']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_mask(self):
  pairs=[(a,b) for a in range(64) for b in range(64) if a>=b and a<53 and b<53];self.assertEqual(len(pairs),53*54//2)
 def test_address(self):
  self.assertEqual([x for p in range(4) for x in range(p*32,(p+1)*32)],list(range(128)));self.assertEqual(1696*32*128,6946816)
 def test_scale(self):self.assertEqual(0.08838834764831845,128**-0.5)

 def test_only_carry_delta(self):
  P=Path(__file__).parent.parent
  old=(P/'tileir_gdn_chunk_output_v1/kernels.py').read_text()
  old=old.replace('bo=ct.add(bo,ct.mma(q,ct.transpose(h),zO),rounding_mode=RM.RN)','bo=ct.mma(q,ct.transpose(h),bo)').replace('ba=ct.add(ba,ct.mma(q,k,zA),rounding_mode=RM.RN)','ba=ct.mma(q,k,ba)')
  self.assertEqual(old,(P/'tileir_gdn_chunk_output_v2/kernels.py').read_text())
 def test_carry_arguments(self):
  s=(Path(__file__).parent.parent/'tileir_gdn_chunk_output_v2/kernels.py').read_text()
  self.assertIn('for block in range(4)',s);self.assertIn('bo=ct.mma(q,ct.transpose(h),bo)',s);self.assertIn('ba=ct.mma(q,k,ba)',s)
