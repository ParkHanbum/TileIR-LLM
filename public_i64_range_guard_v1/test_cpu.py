import unittest
from .contract import descriptor,reference,MIN,MAX
class Tests(unittest.TestCase):
 def test_bounds(self):
  self.assertTrue(reference([MIN,MAX-1],MIN,MAX));self.assertFalse(reference([MAX],MIN,MAX));self.assertTrue(reference([],1,0))
 def test_offsets(self):
  self.assertEqual(descriptor((32,6),(6,1),'torch.int64',3,195*8,0,248320),192)
 def test_stride(self):
  with self.assertRaises(ValueError):descriptor((32,),(2,),'torch.int64',0,64*8,0,32)
 def test_empty(self):self.assertEqual(descriptor((0,),(1,),'torch.int64',2,16,0,1),0)
 def test_bounds_errors(self):
  for lo,hi in [(MIN-1,0),(0,MAX+1),(False,1)]:
   with self.assertRaises(ValueError):reference([],lo,hi)
 def test_backing(self):
  with self.assertRaises(ValueError):descriptor((32,),(1,),'torch.int64',1,256,0,32)
 def test_shortcircuit(self):
  seen=[]
  def later():seen.append(1);raise RuntimeError()
  self.assertFalse(reference([-1],0,32)and later());self.assertEqual(seen,[])
