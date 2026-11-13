import unittest,json
from pathlib import Path
from .run import validate_manifest,result_gate
class Tests(unittest.TestCase):
 def test_actual(self):
  p=Path(__file__).parent.parent/'attention_core_runtime_capture_v1/attentionCoreCapture01/bank/MANIFEST.json';self.assertEqual(len(validate_manifest(json.loads(p.read_text()))),2)
 def test_partial(self):self.assertFalse(result_gate({'complete':False,'records':[]})['raw_pass'])
 def test_hold(self):self.assertTrue(result_gate({'raw_difference_hold':True})['raw_difference_hold'])
