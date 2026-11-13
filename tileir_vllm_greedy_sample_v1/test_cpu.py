import unittest
from .api import reference_rows,validate_descriptor
class T(unittest.TestCase):
 def test_masks(self):self.assertEqual(reference_rows([[float('-inf'),2,1],[1,float('-inf'),3]]),[1,2])
 def test_ties(self):self.assertEqual(reference_rows([[2,2,-1],[-0.,0.]]),[0,0])
 def test_invalid(self):
  for row in [[float('nan'),1],[float('inf'),1],[float('-inf')]*3]:
   with self.assertRaises(AssertionError):reference_rows([row])
 def test_descriptor(self):
  d=dict(shape=[32,248320],stride=[248320,1],storage_offset=0,dtype='torch.float32',requires_grad=False);validate_descriptor(d)
  with self.assertRaises(AssertionError):validate_descriptor(dict(d,storage_offset=1))
