import unittest,tempfile,shutil
from pathlib import Path
from attention_kv_runtime_integration_v2.preflight import preflight
class Tests(unittest.TestCase):
 def test_real(self):self.assertEqual(preflight(Path(__file__).parent)['expected_tokens'],512)
 def test_missing_and_corrupt(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for name in ('run.py','launch.py','adapter.py','gates.py','result_gate.py','precision.py','EXPECTED-TOKENS.json'):shutil.copyfile(Path(__file__).parent/name,root/name)
   (root/'EXPECTED-TOKENS.json').unlink()
   with self.assertRaises(AssertionError):preflight(root)
   (root/'EXPECTED-TOKENS.json').write_text('null')
   with self.assertRaises(AssertionError):preflight(root)
