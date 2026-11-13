import unittest,copy,ast
from pathlib import Path
from .api import validate_metadata

def fixture():
 def m(shape,stride,dtype):return dict(shape=shape,stride=stride,dtype='torch.'+dtype,offset=0)
 return [m((8192,1696),(1,12288),'bfloat16'),m((311,8192,3),(1081344,1,8192),'bfloat16'),m((8192,4),(4,1),'bfloat16'),m((33,),(1,),'int32'),m((32,),(1,),'int32'),m((32,),(1,),'bool'),list(range(0,1697,53)),list(range(128,0,-4)),[False]*32]
class Contract(unittest.TestCase):
 def test_actual_metadata(self):self.assertTrue(validate_metadata(*fixture()))
 def test_initial_state_rejected(self):
  a=fixture();a[-1][0]=True
  with self.assertRaises(ValueError):validate_metadata(*a)
 def test_boundaries_rejected(self):
  a=fixture();a[6][1]=52
  with self.assertRaises(ValueError):validate_metadata(*a)
 def test_duplicate_rejected(self):
  a=fixture();a[7][1]=a[7][0]
  with self.assertRaises(ValueError):validate_metadata(*a)
 def test_null_rejected(self):
  a=fixture();a[7][0]=0
  with self.assertRaises(ValueError):validate_metadata(*a)
 def test_bools_not_slots(self):
  a=fixture();a[7][0]=True
  with self.assertRaises(ValueError):validate_metadata(*a)
 def test_bad_layouts(self):
  for index,key,value in [(0,'stride',(2,12288)),(1,'stride',(100,1,8192)),(1,'shape',(311,8192,4)),(2,'dtype','torch.float32'),(3,'dtype','torch.int64'),(4,'offset',-1)]:
   a=fixture();a[index][key]=value
   with self.assertRaises(ValueError):validate_metadata(*a)
 def test_tensor_shapes(self):
  a=fixture();a[5]['shape']=(31,)
  with self.assertRaises(ValueError):validate_metadata(*a)
 def test_source_syntax(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
