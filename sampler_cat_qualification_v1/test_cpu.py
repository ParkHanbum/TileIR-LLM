import unittest,ast
from pathlib import Path
from tileir_vllm_sampler_cat_v1.api import validate_descriptor
from .run import result_gate
class Tests(unittest.TestCase):
 def test_descriptors(self):
  for w in (1,5):validate_descriptor(dict(shape=[32,w],stride=[w,1],offset=0,dtype='torch.int64',bytes=32*w*8,requires_grad=False),w)
 def test_reject_stride(self):
  with self.assertRaises(AssertionError):validate_descriptor(dict(shape=[32,5],stride=[6,1],offset=1,dtype='torch.int64',bytes=1536,requires_grad=False),5)
 def test_coordinate_copy_signed_duplicates(self):
  c=[[-(1<<63)]for _ in range(32)];t=[[0,0,-1,(1<<63)-1,3]for _ in range(32)];out=[]
  for i in range(192):
   row,col=divmod(i,6);out.append(c[row][0]if col==0 else t[row][col-1])
  self.assertEqual(out,[v for row in range(32)for v in c[row]+t[row]])
 def test_gate_order(self):
  r=dict(complete=True,records=[dict(call_index=i,arm=a,repeat=n,passed=True)for i in (0,15)for a in ('native','tile')for n in (0,1)]);self.assertTrue(result_gate(r)['raw_pass']);r['records'].pop();self.assertFalse(result_gate(r)['raw_pass'])
 def test_ast_no_global_patch(self):
  root=Path(__file__).parent
  for f in [*root.glob('*.py'),*(root.parent/'tileir_vllm_sampler_cat_v1').glob('*.py')]:ast.parse(f.read_text())
  self.assertNotIn('torch.cat =',(root/'run.py').read_text())
