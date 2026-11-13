import ast,unittest
from pathlib import Path
from contract import validate_metadata,validate_descriptor
class Tests(unittest.TestCase):
 def test_metadata(self):self.assertEqual(validate_metadata(list(range(0,1697,53)),[[i,0] for i in range(32)]),32)
 def test_order_rejected(self):
  with self.assertRaises(AssertionError):validate_metadata(list(range(0,1697,53)),[[31-i,0] for i in range(32)])
 def test_length_rejected(self):
  with self.assertRaises(AssertionError):validate_metadata(list(range(33)),[[i,0] for i in range(32)])
 def test_descriptor(self):validate_descriptor((1,1696,32),(54272,32,1),'torch.float32',0,64,False,False)
 def test_bad_options(self):
  for chunk,reverse,head in [(32,False,False),(True,False,False),(64,True,False),(64,False,True)]:
   with self.assertRaises(AssertionError):validate_descriptor((1,1696,32),(54272,32,1),'torch.float32',0,chunk,reverse,head)
 def test_bad_stride(self):
  with self.assertRaises(AssertionError):validate_descriptor((1,1696,32),(54272,1,1696),'torch.float32',0,64,False,False)
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
