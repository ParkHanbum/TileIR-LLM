import unittest,tempfile,ast
from pathlib import Path
from .adapter import extent,values_contract,advance,Adapter,ORDER
from .result_gate import audit
R=Path(__file__).parent
class Tests(unittest.TestCase):
 def test_extent_gap_and_offset(self):
  e=extent((310,32,128,128),(540672,16384,128,1),128,4,1024**3);self.assertGreater(e['storage_extent_bytes'],e['logical_bytes']);self.assertEqual(e['storage_offset_bytes'],512)
 def test_extent_rejects_before_copy(self):
  for shape,stride,storage in [((10000,32,128,128),(540672,16384,128,1),64*1024**3),((310,32,128,128),(540672,16384,128,1),1024)]:
   with self.assertRaises(AssertionError):extent(shape,stride,0,4,storage)
 def test_indices_bounds_duplicates_mixedmask(self):
  a=values_contract(list(range(32)),[i%2==0 for i in range(32)],310);self.assertTrue(a['unique'])
  a=values_contract([-1]*32,[False]*32,310);self.assertFalse(a['unique']);self.assertEqual(a['normalized_indices'],[309]*32)
  with self.assertRaises(AssertionError):values_contract([310]*32,[False]*32,310)
 def test_exact_order_first_failure(self):
  row={'order':[]}
  for x in ORDER:advance(row,x)
  with self.assertRaises(AssertionError):advance(row,'gather')
  with self.assertRaises(AssertionError):advance({'order':[]},'scatter')
 def test_latch(self):
  with tempfile.TemporaryDirectory()as d:
   a=Adapter(Path(d)/'out');a.fail(ValueError('first'));f=a.failed.copy();a.fail(ValueError('second'));self.assertEqual(a.failed,f)
   with self.assertRaises(AssertionError):a.start('shadow')
 def test_no_owner_copy_or_numeric_port(self):
  s=(R/'adapter.py').read_text();self.assertNotIn('owner.clone',s);self.assertNotIn('owner.cpu',s);self.assertNotIn('ct.launch',s);self.assertIn('t.numel()*t.element_size()<=256',s)
 def test_ast_and_gate(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
  self.assertFalse(audit({})['passed'])
