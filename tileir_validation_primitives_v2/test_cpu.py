import unittest,math,ast,json
from pathlib import Path
from unittest.mock import patch
from .contract import checked_status,validate_finite_descriptor
class T(unittest.TestCase):
 def test_flags_accepted(self):validate_finite_descriptor((32,),(1,),'torch.float32',0,False,True,True)
 def test_cpu_real_alias(self):
  import torch
  with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA forbidden'))as guard:
   for dtype in (torch.float32,torch.bfloat16):
    x=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),1e-40],dtype=dtype)
    for view in (x,x.conj()):
     raw=torch.empty(0,dtype=view.dtype).set_(view.untyped_storage(),view.storage_offset(),(view.numel(),),(1,));self.assertFalse(raw.is_neg());self.assertFalse(raw.is_conj());self.assertEqual(raw.data_ptr(),view.data_ptr());self.assertTrue(torch.equal(torch.isfinite(raw),torch.isfinite(view)))
   z=torch.complex(torch.tensor([0.,-0.,float('inf'),float('nan')]),torch.ones(4));v=z.conj().imag.as_strided((8,),(1,),0);self.assertTrue(v.is_neg());before=z.view(torch.uint8).clone();raw=torch.empty(0,dtype=v.dtype).set_(v.untyped_storage(),0,(8,),(1,));self.assertFalse(raw.is_neg());self.assertTrue(torch.equal(torch.isfinite(raw),torch.isfinite(v)));self.assertTrue(torch.equal(before,z.view(torch.uint8)));guard.assert_not_called()
 def test_failure_before_target(self):
  calls=[]
  def target(flag):checked_status(flag);calls.append(1)
  with self.assertRaises(AssertionError):target(1)
  self.assertEqual(calls,[]);target(0);self.assertEqual(calls,[1])
 def test_active_binding(self):
  d=json.loads(Path(__file__).with_name('ACTIVE-PATH.json').read_text())['observed_component'];self.assertEqual(d['suboperation_counts']['top5'],16)
 def test_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
