import unittest
from api import validate_metadata
def spec(shape,dtype='torch.bfloat16',stride=None,offset=0):return dict(shape=shape,stride=stride or ([shape[1],1] if len(shape)==2 else [1]),dtype=dtype,offset=offset)
def args():return [spec([32,8192],stride=[10240,1]),spec([32,1024],stride=[10240,1],offset=8192),spec([256]),spec([256]),spec([262144,64]),spec([3,32],'torch.int64',[2049,1]),1e-6,16,4,256,64,[11,11,10],1.0]
class Tests(unittest.TestCase):
 def test_actual(self):self.assertEqual(validate_metadata(*args()),32)
 def test_strides(self):
  a=args();a[0]['stride']=[8191,1]
  with self.assertRaises(ValueError):validate_metadata(*a)
 def test_positions(self):
  a=args();a[5]['stride']=[31,1]
  with self.assertRaises(ValueError):validate_metadata(*a)
 def test_scope(self):
  a=args();a[-1]=0.0
  with self.assertRaises(NotImplementedError):validate_metadata(*a)
 def test_dtype(self):
  a=args();a[2]['dtype']='torch.float32'
  with self.assertRaises(ValueError):validate_metadata(*a)
if __name__=='__main__':unittest.main()
