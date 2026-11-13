import unittest,json,copy
from pathlib import Path
from .run import check
from .launch import result_gate
P=Path(__file__).resolve().parent.parent
class Test(unittest.TestCase):
 def test_actual(self):self.assertTrue(check(json.loads((P/'tp1_embedding_runtime_capture_v2/tp1EmbeddingCapture01/bank/MANIFEST.json').read_text())))
 def test_bad_stride(self):
  m=json.loads((P/'tp1_embedding_runtime_capture_v2/tp1EmbeddingCapture01/bank/MANIFEST.json').read_text());m['samples'][0]['input']['metadata']['stride']=[2]
  with self.assertRaises(AssertionError):check(m)
 def test_incomplete(self):self.assertFalse(result_gate({'complete':False})['raw_pass'])
 def test_failed_call(self):
  c=dict(phase='prefill',input_unchanged=True,reference_repeat_raw=True,candidate_repeat_raw=True,finite=True,raw_equal=True,calls=[dict(raw_equal=True,finite=True,strict=True,fresh=False,input_unchanged=True)]*4)
  d=copy.deepcopy(c);d['phase']='decode'
  self.assertFalse(result_gate(dict(complete=True,gpu_executed=True,records=[c,d]))['raw_pass'])
