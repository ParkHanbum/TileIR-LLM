import unittest,sys,runpy
from pathlib import Path
from unittest.mock import patch
from .analyze import analyze
from .admission import require_ready
R=Path(__file__).parent
class Tests(unittest.TestCase):
 def events(self):
  e=[dict(ph='X',cat='user_annotation',name='request_'+p+'::lean',pid=1,tid=1,ts=i*100,dur=90)for i,p in enumerate(('install','execution','proof','restore'))]
  e += [dict(ph='X',cat='cuda_runtime',name='cudaLaunchKernel',pid=1,tid=1,ts=110,dur=1,args={'correlation':1}),dict(ph='X',cat='kernel',name='s',pid=2,tid=2,ts=999,dur=3,args={'correlation':1})]
  return e
 def test_actual_entry_pending(self):
  with patch.object(sys,'argv',[str(R/'run.py'),'--run-gpu']),patch.object(sys,'path',[str(R)]+sys.path),patch('whole_lean_profile_prepare_v4.admission.require_ready',side_effect=RuntimeError('v7 pending')):
   with self.assertRaisesRegex(RuntimeError,'v7'):runpy.run_path(str(R/'run.py'),run_name='__main__')
  for name in ('torch','vllm','cuda.tile'):self.assertNotIn(name,sys.modules)
 def test_two_evidence_levels(self):
  symbol=dict(verified_own_elf=True,artifact_sha256='artifact',source_file='public.py',entry='s',source_sha256='source')
  r=analyze(self.events(),'lean',{'s':symbol});self.assertEqual(r['compute_count'],{'tileir_source_symbol_matched':1});self.assertTrue(r['compute_all_source_symbol_matched_tileir']);self.assertFalse(r['loaded_artifact_all_verified']);self.assertFalse(r['all_gpu_work_converted_claim'])
 def test_h2h_visible_and_role_unknown(self):
  e=self.events()+[dict(ph='X',cat='gpu_memcpy',name='Memcpy HtoH (Pinned -> Pageable)',pid=2,tid=2,ts=999,dur=1,args={'correlation':1})]
  r=analyze(e,'lean');self.assertEqual(r['totals']['execution/memcpy_H2H']['count'],1);self.assertEqual(r['functional_noncompute_review'][0]['source_role'],'unknown')
 def test_only_direction_extension(self):
  original=(R.parent/'whole_lean_profile_prepare_v2/analyze.py').read_text().replace("('htod','dtoh','dtod')","('htod','dtoh','dtod','htoh')").replace("'dtod':'D2D'","'dtod':'D2D','htoh':'H2H'")
  self.assertEqual(original,(R/'analyze.py').read_text())
if __name__=='__main__':unittest.main()
