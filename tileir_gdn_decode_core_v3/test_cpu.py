import unittest
from api import validate_metadata
class Tests(unittest.TestCase):
 def fixture(self):
  shapes={'mixed_qkv':((32,8192),(12288,1),'torch.bfloat16'),'a':((32,32),(32,1),'torch.bfloat16'),'b':((32,32),(32,1),'torch.bfloat16'),'A_log':((32,),(1,),'torch.float32'),'dt_bias':((32,),(1,),'torch.bfloat16'),'initial_state':((310,32,128,128),(540672,16384,128,1),'torch.float32'),'out':((32,1,32,128),(4096,4096,128,1),'torch.bfloat16'),'ssm_state_indices':((32,),(1,),'torch.int32')}
  return {k:dict(shape=v[0],stride=v[1],dtype=v[2],offset=12288 if k=='initial_state'else 0)for k,v in shapes.items()},list(range(4,129,4))
 def test_actual(self):self.assertTrue(validate_metadata(*self.fixture(),128**-0.5,True))
 def test_null(self):
  m,s=self.fixture();s[0]=-1;self.assertTrue(validate_metadata(m,s,128**-0.5,True))
 def test_duplicate(self):
  m,s=self.fixture();s[1]=s[0]
  with self.assertRaises(ValueError):validate_metadata(m,s,128**-0.5,True)
 def test_axis(self):
  m,s=self.fixture();m['initial_state']['stride']=(540672,16384,1,128)
  with self.assertRaises(ValueError):validate_metadata(m,s,128**-0.5,True)
 def test_scale(self):
  with self.assertRaises(ValueError):validate_metadata(*self.fixture(),1.0,True)
if __name__=='__main__':unittest.main()
