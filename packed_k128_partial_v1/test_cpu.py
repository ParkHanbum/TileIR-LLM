import ast,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import api
class Tests(unittest.TestCase):
 def test_import_no_backend(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda.tile',sys.modules)
 def test_exact_scope(self):
  self.assertEqual(api.selected_path(32,64,2560),'bf16_k128')
  for dims in [(1696,64,2560),(32,12288,2560),(1696,18432,2560),(32,64,128),(64,64,2560)]:self.assertEqual(api.selected_path(*dims),'unchanged_base')
  with self.assertRaises(TypeError):api.selected_path(True,64,2560)
 def test_ranges_no_gaps_overlap(self):
  parts=api.tile_ranges(2560);self.assertEqual(len(parts),20);self.assertEqual([t for p in parts for t in p],list(range(80)))
  ks=[32*t+16*h+q for p in parts for t in p for h in (0,1) for q in range(16)];self.assertEqual(ks,list(range(2560)))
 def test_reject_bad_block(self):
  for k in [0,2559,64,True]:
   with self.assertRaises(ValueError):api.tile_ranges(k)
 def test_target_route(self):
  x,w,y=object(),object(),object()
  with patch.object(api,'_validate',return_value=SimpleNamespace(m=32,n=64,k=2560)),patch.object(api,'_allocate',return_value=y)as allocate,patch.object(api,'_enqueue')as enqueue,patch.object(api,'_base')as base:
   self.assertIs(api.linear(x,w),y);enqueue.assert_called_once_with(x,w,y);base.assert_not_called();allocate.assert_called_once()
 def test_out_identity(self):
  x,w,y=object(),object(),object()
  with patch.object(api,'_validate',return_value=SimpleNamespace(m=32,n=64,k=2560)),patch.object(api,'_allocate')as allocate,patch.object(api,'_enqueue')as enqueue:
   self.assertIs(api.linear(x,w,out=y),y);allocate.assert_not_called();enqueue.assert_called_once_with(x,w,y)
 def test_other_shape_exact_base_route(self):
  x,w,out,ret=object(),object(),object(),object()
  with patch.object(api,'_validate',return_value=SimpleNamespace(m=1696,n=64,k=2560)),patch.object(api,'_base',return_value=ret)as base,patch.object(api,'_enqueue')as enqueue:
   self.assertIs(api.linear(x,w,out),ret);base.assert_called_once_with(x,w,out);enqueue.assert_not_called()
 def test_source_ast(self):
  R=Path(__file__).parent
  for p in R.glob('*.py'):ast.parse(p.read_text())
  tree=ast.parse((R/'kernels.py').read_text());calls=[x for x in ast.walk(tree)if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)]
  self.assertEqual(sum(x.func.attr=='mma'for x in calls),2);self.assertEqual(sum(x.func.attr=='astype'for x in calls),3);self.assertFalse(any(isinstance(x,ast.If)for x in ast.walk(tree)))
if __name__=='__main__':unittest.main()
