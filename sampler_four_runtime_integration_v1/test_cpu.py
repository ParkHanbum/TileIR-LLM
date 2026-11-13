import unittest,tempfile,ast,json
from pathlib import Path
from unittest.mock import patch
from .adapter import Adapter
from .result_gate import audit
class Tests(unittest.TestCase):
 def setUp(self):
  self.meta_patch=patch('sampler_four_runtime_integration_v1.adapter.meta',side_effect=lambda t:dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),storage_offset=t.storage_offset(),storage_nbytes=t.untyped_storage().nbytes()))
  self.meta_patch.start();self.addCleanup(self.meta_patch.stop)
  self.cuda_patch=patch('torch.cuda._lazy_init',side_effect=AssertionError('CPU test may not initialize CUDA'))
  self.cuda_barrier=self.cuda_patch.start();self.addCleanup(self.cuda_patch.stop)
 def tearDown(self):self.cuda_barrier.assert_not_called()

 def test_copy_only_cat_delta(self):
  p=Path(__file__).parent;old=(p.parent/'sampler_three_ordered_runtime_integration_v2/copied_scores.py.txt').read_text();new=(p/'copied_scores.py.txt').read_text();self.assertEqual(new,old.replace('torch.cat((logprob_token_ids, topk_indices), dim=1)','_tileir_cat(logprob_token_ids, topk_indices)'))
 def test_ast(self):
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
 def test_no_cache_copied(self):
  # Construction whitelist evidence, independent of later parent cache provisioning.
  s=(Path(__file__).parent/'PLAN.md').read_text();self.assertIn('no cache, historical receipt',s)
 def test_partial_gate(self):self.assertFalse(audit({})['passed'])
 def test_cat_exact_both_modes(self):
  import torch
  with tempfile.TemporaryDirectory() as td:
   a=Adapter(Path(td)/'a');a.pending={};c=torch.arange(32,dtype=torch.int64).reshape(32,1);t=torch.arange(160,dtype=torch.int64).reshape(32,5);expected=torch.cat((c,t),1)
   with patch('tileir_vllm_sampler_cat_v1.concat_sampled_top5',side_effect=lambda c,t:torch.cat((c,t),1)):
    out=a.cat(c,t);self.assertTrue(torch.equal(out,expected));self.assertEqual(a.cat_native_calls,1);self.assertTrue(a.cat_rows[0]['passed'])
    a.mode='replace';a.rows=[];a.cat_rows=[];a.cat_native_calls=0;a.cat_tile_calls=0
    out=a.cat(c,t);self.assertTrue(torch.equal(out,expected));self.assertEqual(a.cat_native_calls,0);self.assertEqual(a.cat_tile_calls,1)
 def test_cat_failure_stops_queued(self):
  import torch
  with tempfile.TemporaryDirectory() as td:
   a=Adapter(Path(td)/'a');a.pending={};c=torch.ones((32,1),dtype=torch.int64);t=torch.ones((32,5),dtype=torch.int64)
   with patch('tileir_vllm_sampler_cat_v1.concat_sampled_top5',return_value=torch.zeros((32,6),dtype=torch.int64))as fn:
    with self.assertRaises(AssertionError):a.cat(c,t)
    with self.assertRaises(AssertionError):a.cat(c,t)
    self.assertEqual(fn.call_count,1);self.assertTrue((a.output/'first_cat_mismatch/values.pt').is_file())
 def test_cat_exception_latches(self):
  import torch
  with tempfile.TemporaryDirectory() as td:
   a=Adapter(Path(td)/'a');a.pending={}
   with patch('tileir_vllm_sampler_cat_v1.concat_sampled_top5',side_effect=RuntimeError('fake')):
    with self.assertRaises(RuntimeError):a.cat(torch.ones((32,1),dtype=torch.int64),torch.ones((32,5),dtype=torch.int64))
    self.assertTrue(a.failed)
    with self.assertRaises(AssertionError):a.start('shadow')
 def test_numeric_apis_unchanged(self):
  s=(Path(__file__).parent/'adapter.py').read_text()
  for name in ('tileir_vllm_top5_v2','tileir_vllm_selected_ranks_v1','tileir_vllm_selected_logprobs_v2'):self.assertIn(name,s)
  self.assertNotIn('torch.cat =',s)
