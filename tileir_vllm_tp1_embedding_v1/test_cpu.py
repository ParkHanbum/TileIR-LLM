import unittest
from .api import validate_metadata
class Tests(unittest.TestCase):
 def inputs(self,n=32):return dict(shape=(n,),stride=(1,),dtype='torch.int64',offset=0),dict(shape=(248320,2560),stride=(2560,1),dtype='torch.bfloat16',offset=0)
 def test_shapes(self):
  for n in (32,1696):self.assertEqual(validate_metadata(*self.inputs(n),[1]*n),(n,2560))
 def test_duplicate_and_bounds(self):self.assertEqual(validate_metadata(*self.inputs(),[0,248319]*16),(32,2560))
 def test_oob(self):
  for v in (-1,248320,True):
   with self.assertRaises(ValueError):validate_metadata(*self.inputs(),[v]*32)
 def test_layout(self):
  i,w=self.inputs();i['stride']=(2,)
  with self.assertRaises(ValueError):validate_metadata(i,w,[1]*32)
 def test_type(self):
  i,w=self.inputs();w['dtype']='torch.float32'
  with self.assertRaises(ValueError):validate_metadata(i,w,[1]*32)
 def test_no_empty(self):
  with self.assertRaises(ValueError):validate_metadata(*self.inputs(),[])
if __name__=='__main__':unittest.main()
