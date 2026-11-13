import unittest,ast,sys
from pathlib import Path
from .contract import descriptor,logical_address
class T(unittest.TestCase):
 def test_bounds_and_offsets(self):
  for n in (32,1696):
   self.assertEqual(descriptor((n,4,256),(10240,256,1),'torch.bfloat16',9216,n*20480),n)
   self.assertEqual(logical_address(n-1,3,255),n*10240-1)
   self.assertEqual(logical_address(0,0,0),9216)
   self.assertLess(n*1024,2**31)
   self.assertEqual(1<<(n-1).bit_length(),32 if n==32 else 2048)
 def test_rejections(self):
  for stride,off,storage in [((1024,256,1),9216,655360),((10240,256,1),0,655360),((10240,256,1),9216,655358)]:
   with self.assertRaises(AssertionError):descriptor((32,4,256),stride,'torch.bfloat16',off,storage)
 def test_selected_not_padding(self):
  # CPU address model only, not tensor execution.
  selected={logical_address(r,h,c)for r in range(32)for h in range(4)for c in range(256)}
  self.assertEqual(len(selected),32768)
  self.assertNotIn(0,selected);self.assertNotIn(9215,selected);self.assertNotIn(10240,selected)
  self.assertIn(10239,selected);self.assertIn(327679,selected)
 def test_source_only(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
  self.assertFalse(any(k in sys.modules for k in ('torch','vllm','cuda.tile')))
if __name__=='__main__':unittest.main()
