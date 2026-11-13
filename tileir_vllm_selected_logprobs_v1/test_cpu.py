import math,unittest
from .api import reference_rows,validate_descriptors
class T(unittest.TestCase):
 def test_duplicate_ids(self):
  r=reference_rows([[0,0]],[[1,1,0]])[0];self.assertEqual(r,[-math.log(2)]*3)
 def test_shift(self):self.assertEqual(reference_rows([[2,1]],[[0,1]]),reference_rows([[102,101]],[[0,1]]))
 def test_invalid(self):
  for rows,ids in [([[float('nan')]],[[0]]),([[float('-inf')]],[[0]]),([[0]],[[-1]])]:
   with self.assertRaises(AssertionError):reference_rows(rows,ids)
 def test_layout(self):
  x=dict(shape=[32,248320],stride=[248320,1],storage_offset=0,dtype='torch.bfloat16',requires_grad=False);ids=dict(shape=[32,6],stride=[6,1],storage_offset=0,dtype='torch.int64',requires_grad=False);validate_descriptors(x,ids)
  with self.assertRaises(AssertionError):validate_descriptors(x,dict(ids,shape=[32,5]))
