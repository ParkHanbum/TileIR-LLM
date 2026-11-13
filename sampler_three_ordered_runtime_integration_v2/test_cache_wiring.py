import ast,unittest
from pathlib import Path
class Wiring(unittest.TestCase):
 def test_gate_before_sensitive_entries(self):
  root=Path(__file__).parent
  for name,boundary in [('run.py','import torch,vllm'),('launch.py',"spec=importlib.util.spec_from_file_location")]:
   s=(root/name).read_text();self.assertLess(s.index('verify_cache(Path(__file__).resolve().parent)'),s.index(boundary))
 def test_adapter_unchanged(self):
  root=Path(__file__).parent;self.assertEqual((root/'adapter.py').read_bytes(),(root.parent/'sampler_three_ordered_runtime_integration_v1/adapter.py').read_bytes())
