import unittest,ast
from pathlib import Path
from types import SimpleNamespace as NS
from .spy import launch_spy
class T(unittest.TestCase):
 def test_spy_restore(self):
  calls=[];old=lambda *a:calls.append(a);ct=NS(launch=old);target=object();guard=object()
  with launch_spy(ct,(target,),(guard,))as c:ct.launch(None,None,guard,());self.assertEqual(c,dict(target=0,guard=1,other=0))
  self.assertIs(ct.launch,old);self.assertEqual(len(calls),1)
 def test_candidate_delta(self):
  P=Path(__file__).resolve().parent.parent;s=(P/'tileir_vllm_top5_v2/api.py').read_text();candidate=Path(__file__).with_name('candidate.py').read_text();expected=s.replace('from .reference import reference','from tileir_vllm_top5_v2.reference import reference').replace('from .kernels import blocks,finish','from tileir_vllm_top5_v2.kernels import blocks,finish').replace(" assert bool(torch.isfinite(logits).all()),'NaN/Inf outside finite-only contract'"," from tileir_validation_primitives_v2.api import require_finite\n require_finite(logits)");self.assertEqual(candidate,expected)
 def test_actual_candidate_path_invalid_no_target_or_alloc(self):
  import sys,types
  from unittest.mock import patch
  from . import candidate
  calls=[]
  class Tensor:
   shape=(32,248320);dtype='torch.bfloat16';requires_grad=False;device=NS(type='cuda')
   def stride(self):return (248320,1)
   def storage_offset(self):return 0
  def forbidden(*a,**kw):calls.append(1);raise AssertionError('Unexpected target allocation/launch')
  def check(x):raise AssertionError('NaN/Inf outside finite-only contract')
  torch=NS(Tensor=Tensor,empty=forbidden);tile=NS(launch=forbidden)
  fake={'torch':torch,'cuda':NS(tile=tile),'cuda.tile':tile,'tileir_vllm_top5_v2.kernels':NS(blocks=object(),finish=object()),'tileir_validation_primitives_v2.api':NS(require_finite=check)}
  with patch.dict(sys.modules,fake):
   with self.assertRaisesRegex(AssertionError,'NaN/Inf outside finite-only contract'):candidate.top5_indices(Tensor())
  self.assertEqual(calls,[])
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
