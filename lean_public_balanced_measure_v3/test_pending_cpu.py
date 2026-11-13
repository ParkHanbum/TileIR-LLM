"""Only stdlib pending-release and mapping tests; no CUDA/model import."""
import ast,json,tempfile,unittest
from pathlib import Path
from . import gates,scopes
from lean_public_composition_v1.support import Base
from lean_public_composition_v2.adapter import SPECS,ORDER
R=Path(__file__).resolve().parent
class PendingTests(unittest.TestCase):
 def test_pending_stops_before_evidence_access(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);(p/'QUALIFICATION-RECEIPT.json').write_text(json.dumps(dict(complete=False,result={'path':'/must/not/read'})))
   with self.assertRaisesRegex(AssertionError,'Lean25 actual correctness pending'):gates.qualification(p)
 def test_old18_or_uncertain19_does_not_qualify(self):
  for package in ('lean_public_runtime_qualification_v1','old19'):
   with tempfile.TemporaryDirectory()as d:
    p=Path(d);(p/'QUALIFICATION-RECEIPT.json').write_text(json.dumps(dict(complete=True,qualification_package=package)))
    with self.assertRaises(AssertionError):gates.qualification(p)
 def test_25_mapping_shared_actual_base(self):
  self.assertEqual(len(SPECS),25);self.assertEqual(sum(n for k,c,n in SPECS),5888)
  self.assertTrue(all(issubclass(c,Base)for k,c,n in SPECS));self.assertEqual(set(ORDER),set(k for k,c,n in SPECS))
 def test_launcher_prerequisite_before_cache_and_admission(self):
  s=(R/'launch.py').read_text();self.assertLess(s.index('qualification(S)'),s.index('cache_preseed=preflight(R)'));self.assertLess(s.index('qualification(S)'),s.index('spec.loader.exec_module(h)'));self.assertLess(s.index('qualification(S)'),s.index('fcntl.flock('))
 def test_worker_prerequisite_before_lock_and_model_import(self):
  s=(R/'run.py').read_text();gate=s.index("result['qualification_receipt']=qualification(R)")
  for after in ('held=os.fstat(',"result['cache_preseed']=preflight(C)",'import torch,vllm'):self.assertLess(gate,s.index(after))
 def test_timer_and_profile_contract_unchanged(self):
  old=R.parent/'lean_public_balanced_measure_v2'
  for file in ('design.py','profile_contract.py'):
   self.assertEqual((R/file).read_bytes(),(old/file).read_bytes())
  before=(old/'request.py').read_text().replace('lean_public_runtime_qualification_v1','lean_public_runtime_qualification_v2')
  self.assertEqual(before,(R/'request.py').read_text())
if __name__=='__main__':unittest.main()
