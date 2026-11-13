import unittest
from .api import reference_rows,validate_descriptors
class T(unittest.TestCase):
 def test_inclusive(self):self.assertEqual(reference_rows([[3,2,2,0],[1,2,3,4]],[1,0]),[3,4])
 def test_mask(self):self.assertEqual(reference_rows([[float('-inf'),0,-1]],[1]),[1])
 def test_ties(self):self.assertEqual(reference_rows([[1,1,1],[-0.,0.,1]],[2,0]),[3,3])
 def test_reject(self):
  for row,i in [([float('nan'),1],1),([float('inf'),1],1),([float('-inf'),1],0),([1],-1)]:
   with self.assertRaises(AssertionError):reference_rows([row],[i])
 def test_descriptor(self):
  x=dict(shape=[32,248320],stride=[248320,1],storage_offset=0,dtype='torch.bfloat16',requires_grad=False);ids=dict(shape=[32],stride=[1],storage_offset=0,dtype='torch.int64',requires_grad=False);validate_descriptors(x,ids)
  with self.assertRaises(AssertionError):validate_descriptors(dict(x,dtype='torch.float32'),ids)
