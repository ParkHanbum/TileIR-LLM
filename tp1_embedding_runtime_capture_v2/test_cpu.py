import unittest
from observer import phase,version
from result_gate import audit
class T(unittest.TestCase):
 def test_phase(self):self.assertEqual(phase(1696),'prefill');self.assertEqual(phase(32),'decode')
 def test_badphase(self):
  with self.assertRaises(AssertionError):phase(1)
 def test_inference_version(self):
  class X:
   @property
   def _version(self):raise RuntimeError('Inference tensors do not track version counter.')
  self.assertEqual(version(X()),'unavailable_inference_tensor')
 def test_other_error(self):
  class X:
   @property
   def _version(self):raise RuntimeError('other')
  with self.assertRaises(RuntimeError):version(X())
 def test_missing_hold(self):self.assertFalse(audit({})['passed'])
if __name__=='__main__':unittest.main()
