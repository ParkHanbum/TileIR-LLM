import unittest
from .contract import validate
class Tests(unittest.TestCase):
 def test_empty(self):self.assertEqual(validate((32,),(1,),'torch.int64',[],[],[],[]),())
 def test_segments(self):self.assertEqual(validate((2,4),(4,1),'torch.int32',[0,1],[1,0],[3,4,5],[2,3]),((1,3),(4,5)))
 def test_repeat_disjoint(self):validate((1,4),(4,1),'torch.int32',[0,0],[0,3],[3,4],[1,2])
 def test_zero_length(self):validate((1,4),(4,1),'torch.int32',[0],[4],[],[0])
 def test_reject(self):
  for args in [([1],[0],[3],[1]),([0],[4],[3],[1]),([0,0],[0,0],[3,4],[1,2]),([0],[0],[3],[0]),([0],[0],[2**31],[1])]:
   with self.assertRaises(AssertionError):validate((1,4),(4,1),'torch.int32',*args)
 def test_float_special(self):validate((1,4),(4,1),'torch.float32',[0],[0],[float('inf'),float('nan'),-0.0],[3])
