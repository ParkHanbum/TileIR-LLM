import unittest
from gdn_decode_core_runtime_v4.adapter import remap,numerical_pass
from gdn_decode_core_runtime_v4.result_gate import audit
class Tests(unittest.TestCase):
 def test_map(self):
  slots=list(range(128,0,-4));s,m=remap(slots,310);self.assertEqual(m,list(range(32,0,-1)));self.assertEqual(s,list(range(4,129,4)))
 def test_null(self):
  with self.assertRaises(AssertionError):remap([0]+list(range(1,32)),310)
 def test_duplicate(self):
  with self.assertRaises(AssertionError):remap([1]*32,310)
 def test_failurestop_source(self):
  import ast
  from pathlib import Path
  tree=ast.parse((Path(__file__).resolve().parent/'adapter.py').read_text())
  wrapper=next(x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.name=='wrapper')
  self.assertIsInstance(wrapper.body[-2],ast.Assert);self.assertIsInstance(wrapper.body[-1],ast.Return)
  self.assertIn('selected_state_finite', (Path(__file__).resolve().parent/'adapter.py').read_text())
 def test_raw_preservation_source(self):
  from pathlib import Path
  s=(Path(__file__).resolve().parent/'adapter.py').read_text()
  self.assertIn('view(torch.uint8)',s);self.assertIn('raw_equal(mapped_i,mapped_before,torch)',s)
 def test_expected(self):
  import copy
  from gdn_decode_core_runtime_v4.expected import load_expected,check_expected
  expected=load_expected();self.assertTrue(check_expected(expected,expected))
  bad=copy.deepcopy(expected);bad[0][0]+=1
  with self.assertRaises(AssertionError):check_expected(bad,expected)
 def test_receipt(self):
  from gdn_decode_core_runtime_v4.gates import qualification
  from pathlib import Path
  self.assertIn('files',qualification(Path(__file__).resolve().parent))
 def test_rawgate(self):
  self.assertFalse(numerical_pass({'state':dict(raw_equal=False,strict_pass=True,candidate_nonfinite=0,reference_nonfinite=0)}));self.assertFalse(audit({})['passed'])
if __name__=='__main__':unittest.main()
