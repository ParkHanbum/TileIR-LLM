import unittest
from api import validate_metadata
class Tests(unittest.TestCase):
 def args(self):
  return [dict(shape=(32,8192),stride=(12288,1),offset=0,dtype='torch.bfloat16'),dict(shape=(311,8192,3),stride=(1081344,1,8192),offset=0,dtype='torch.bfloat16'),dict(shape=(8192,4),stride=(4,1),offset=0,dtype='torch.bfloat16'),dict(shape=(32,),stride=(1,),offset=0,dtype='torch.int32'),list(range(4,129,4))]
 def test_actual(self):self.assertEqual(validate_metadata(*self.args()),32)
 def test_null(self):
  a=self.args();a[-1][0]=0;self.assertEqual(validate_metadata(*a),31)
 def test_duplicate(self):
  a=self.args();a[-1][1]=a[-1][0]
  with self.assertRaises(ValueError):validate_metadata(*a)
 def test_oob(self):
  a=self.args();a[-1][0]=311
  with self.assertRaises(ValueError):validate_metadata(*a)
 def test_overlap(self):
  a=self.args();a[1]['stride']=(8192,1,8192)
  with self.assertRaises(ValueError):validate_metadata(*a)
 def test_bad_dtype(self):
  a=self.args();a[0]['dtype']='torch.float32'
  with self.assertRaises(ValueError):validate_metadata(*a)
 def test_boolslot(self):
  a=self.args();a[-1][0]=True
  with self.assertRaises(ValueError):validate_metadata(*a)
if __name__=='__main__':unittest.main()
