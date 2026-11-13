import ast,unittest
from pathlib import Path
from contract import descriptor,key_head,mask
class Tests(unittest.TestCase):
 def test_descriptor(self):
  for bk in (32,64,128):self.assertEqual(descriptor((1,1696,16,128),(3473408,2048,128,1),(1,1696,32),(54272,32,1),(1,1696,32),(54272,32,1),bk),(1,1696,32,64))
 def test_bad_bk(self):
  for bk in (16,256,True):
   with self.assertRaises(AssertionError):descriptor((1,1696,16,128),(3473408,2048,128,1),(1,1696,32),(54272,32,1),(1,1696,32),(54272,32,1),bk)
 def test_head_map(self):self.assertEqual([key_head(h) for h in range(32)],[h for h in range(16) for _ in range(2)])
 def test_mask_count(self):self.assertEqual(sum(mask(r,c) for r in range(53) for c in range(64)),53*52//2)
 def test_diagonal_padding(self):
  for r in range(53):
   self.assertFalse(mask(r,r))
   for c in range(r,64):self.assertFalse(mask(r,c))
 def test_strides(self):
  with self.assertRaises(AssertionError):descriptor((1,1696,16,128),(3473408,128,2048,1),(1,1696,32),(54272,32,1),(1,1696,32),(54272,32,1),64)
 def test_supported_conversion_expression(self):
  s=(Path(__file__).parent/'kernels.py').read_text()
  self.assertNotIn('bitcast',s)
  self.assertEqual(s.count('astype(ct.tfloat32,rounding_mode=RM.RZ)'),2)
 def test_k8_expression(self):
  s=(Path(__file__).parent/'kernels.py').read_text()
  self.assertIn('for block in range(16):',s)
  self.assertIn('col=(block//2)*16+(block%2)+2*ct.arange(8,dtype=ct.int32)',s)
  self.assertIn('accum=ct.mma(lhs,rhs,accum)',s)
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
class MappingTests(unittest.TestCase):
 def test_permutation(self):
  groups=[[(block//2)*16+(block%2)+2*k for k in range(8)] for block in range(16)]
  self.assertEqual(sorted(sum(groups,[])),list(range(128)))
  self.assertEqual(groups[0],list(range(0,16,2)));self.assertEqual(groups[1],list(range(1,16,2)))
 def test_only_source_delta(self):
  before=(Path(__file__).parent.parent/'tileir_gdn_chunk_kkt_v3/kernels.py').read_text()
  self.assertEqual(before.replace('col=block*8+ct.arange(8,dtype=ct.int32)','col=(block//2)*16+(block%2)+2*ct.arange(8,dtype=ct.int32)'),(Path(__file__).parent/'kernels.py').read_text())
if __name__=='__main__':unittest.main()

