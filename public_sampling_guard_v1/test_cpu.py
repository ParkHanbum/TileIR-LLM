import unittest,ast
from pathlib import Path
from public_sampling_guard_v1.contract import matrix,reference,masks
class Tests(unittest.TestCase):
 def test_orders(self):
  src=(Path(__file__).parent/'api.py').read_text()
  ast.parse(src);self.assertLess(src.index('if not all_in_range_i64'),src.index('ct.launch(torch.cuda.current_stream'))
 def test_classes(self):
  for dtype in ('torch.float32','torch.bfloat16'):
   a,i=masks(dtype);sign=a+1
   for w in (0,sign,1,sign|1,i-1,sign|(i-1)):
    self.assertFalse(reference([[w]],'has_nan',dtype));self.assertFalse(reference([[w]],'has_posinf',dtype));self.assertTrue(reference([[w]],'rows_have_finite',dtype))
   for w in (i+1,sign|(i+1),a,a|sign):self.assertTrue(reference([[w]],'has_nan',dtype))
   self.assertTrue(reference([[i]],'has_posinf',dtype));self.assertFalse(reference([[sign|i]],'has_posinf',dtype))
 def test_negative_inf_rows(self):
  self.assertTrue(reference([[0xff800000,0],[1,0xff800000]],'rows_have_finite','torch.float32'))
  self.assertFalse(reference([[0,1],[0xff800000,0xff800000]],'rows_have_finite','torch.float32'))
 def test_empty(self):
  self.assertTrue(reference([],'rows_have_finite','torch.float32'))
  self.assertFalse(reference([[]],'rows_have_finite','torch.float32'))
  self.assertFalse(reference([],'has_nan','torch.float32'))
 def test_descriptor(self):
  self.assertEqual(matrix((32,248320),(248320,1),'torch.float32',0,32*248320*4),(32,248320,4))
  self.assertEqual(matrix((2,3),(3,1),'torch.bfloat16',2,16),(2,3,2))
 def test_rejection(self):
  for args in [((2,3),(4,1),'torch.float32',0,32),((33,3),(3,1),'torch.float32',0,396),((1,3),(3,1),'torch.float64',0,24),((2,3),(3,1),'torch.float32',1,24)]:
   with self.assertRaises(ValueError):matrix(*args)
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
